"""Optional W&B scalar monitoring. Credentials stay in the SDK environment/netrc."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import uuid
from .provenance import write_json


def public_config(config, provenance):
    """Only experiment settings and identifiers, never credentials or file contents."""
    return {
        "model": {"adapter": config.model.adapter, "initialization": config.model.init},
        "width": asdict(config.width), "loss": asdict(config.loss),
        "training": asdict(config.training), "runtime": asdict(config.runtime),
        "data": {name: getattr(config.data, name) for name in
                 ("frames", "height", "width", "frame_stride", "batch_size", "horizontal_flip")},
        "teacher_sha256": provenance.get("teacher_sha256"),
        "source_commit": provenance.get("source", {}).get("commit"),
        "manifests": provenance.get("data", {}),
    }


class WandbTracker:
    def __init__(self, config, output, provenance=None, state=None, job_type="decoder-recovery"):
        self.config = config
        self.output = Path(output)
        self.state = dict(state or {})
        self.run = None
        w = config.wandb
        if not w.enabled or w.mode == "disabled":
            return
        config.validate()
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B enabled: install decoder-speedup[tracking] in the project environment") from exc
        if self.state and (self.state["entity"] != w.entity or self.state["project"] != w.project):
            raise ValueError("Resume cannot silently move an existing W&B run to another entity/project")
        if self.state and self.state["mode"] != w.mode:
            raise ValueError("Keep W&B mode unchanged when resuming, or disable tracking")
        resume_online = bool(self.state) and w.mode == "online"
        previous_id = self.state.get("run_id")
        run_id = previous_id if resume_online else uuid.uuid4().hex[:8]
        directory = self.output / "wandb-logs"
        directory.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            entity=w.entity, project=w.project, id=run_id,
            name=None if resume_online else (w.name or self.output.name),
            group=w.group or None, tags=w.tags, job_type=job_type,
            mode=w.mode, resume="must" if resume_online else ("never" if w.mode == "online" else None),
            dir=str(directory), config=public_config(config, provenance or {}),
            save_code=False, force=True,
            settings=wandb.Settings(console="off", disable_code=True, disable_git=True, init_timeout=30),
        )
        self.state = {"run_id": self.run.id, "entity": w.entity, "project": w.project, "mode": w.mode}
        try:
            if previous_id and not resume_online:
                # W&B does not resume offline runs. Keep an explicit segment lineage.
                self.run.summary["resumed_from_offline_run"] = previous_id
            self.run.define_metric("progress/generator_updates")
            for pattern in ("train/*", "optim/*", "validation/*", "timing/*", "progress/*"):
                self.run.define_metric(pattern, step_metric="progress/generator_updates")
            write_json({**self.state, "url": self.run.url}, self.output / "wandb-run.json")
        except BaseException:
            self.run.finish(exit_code=1)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            self.run.finish(exit_code=0 if exc_type is None else 1)
        return False

    def log_training(self, metrics, trainer):
        if self.run is None:
            return
        update = metrics["generator_updates"]
        t = self.config.training
        total = t.reconstruction_updates + t.adversarial_updates
        boundary = {1, t.reconstruction_updates, t.reconstruction_updates + 1, total}
        if update % self.config.wandb.log_every and update not in boundary:
            return
        data = {"progress/generator_updates": update,
                "progress/discriminator_updates": metrics["discriminator_updates"],
                "progress/microbatches": metrics["microbatches"],
                "progress/fraction": update / total,
                "progress/epoch": trainer.stream.epoch,
                "progress/dataset_cursor": trainer.stream.cursor,
                "train/stage": metrics["stage"],
                "train/gan_active": int(metrics["stage"] == "adversarial"),
                "optim/generator_lr": trainer.optimizer.param_groups[0]["lr"],
                "timing/update_seconds": metrics["update_seconds"]}
        for name in ("loss", "l1", "lpips", "feature", "g_gan", "gan_weight", "d_loss"):
            if name in metrics:
                data["train/" + name] = metrics[name]
        if trainer.d_optimizer:
            data["optim/discriminator_lr"] = trainer.d_optimizer.param_groups[0]["lr"]
        # W&B's internal step is automatic; charts use the real G-update counter.
        self.run.log(data)

    def log_validation(self, result, updates):
        if self.run is None:
            return
        data = {"progress/generator_updates": updates, "validation/clips": result["clips"]}
        means = result["means"]
        for label in ("teacher", "student"):
            for metric, value in means[label].items():
                data[f"validation/{label}/{metric}"] = value
        for metric in means["student"].keys() & means["teacher"].keys():
            data[f"validation/student_minus_teacher/{metric}"] = means["student"][metric] - means["teacher"][metric]
        self.run.log(data)

    def mark_complete(self, trainer):
        if self.run is not None:
            self.run.summary.update({"training_complete": True,
                                     "generator_updates": trainer.updates,
                                     "discriminator_updates": trainer.d_updates,
                                     "microbatches": trainer.microbatches})


def check_connection(entity, project, output):
    """Create one clearly labelled metrics-only check, without loading a model."""
    from .config import Config, WandbConfig
    config = Config(wandb=WandbConfig(enabled=True, entity=entity, project=project,
                                      name="connection-check", group="setup", tags=["connection-check"]))
    with WandbTracker(config, output, job_type="connection-check") as tracker:
        tracker.run.log({"setup/connection_ok": 1, "setup/optimizer_updates": 0})
        tracker.run.summary.update({"purpose": "W&B connection check only", "formal_training_started": False})
        result = {**tracker.state, "url": tracker.run.url, "connection_ok": True,
                  "formal_training_started": False}
    write_json(result, Path(output) / "connection-check.json")
    return result
