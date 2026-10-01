"""Step-aligned scalar monitoring; no model hooks, media, or GPU synchronization."""
from __future__ import annotations

import copy
from dataclasses import asdict
import time
from pathlib import Path
import uuid
from .config import ORIGINAL_WIDTHS
from .provenance import write_json


SCALAR_GROUPS = {
    "loss": "train", "l1": "train", "lpips": "train", "feature": "train",
    "g_gan": "gan", "gan_weight": "gan", "d_loss": "gan",
    "g_grad_norm": "optim", "d_grad_norm": "optim", "update_seconds": "timing",
}


def public_config(config, provenance):
    """Only experiment settings and identifiers, never credentials or file contents."""
    return {
        "monitoring_schema": 2,
        "model": {"adapter": config.model.adapter, "initialization": config.model.init},
        "width": asdict(config.width), "loss": asdict(config.loss),
        "training": asdict(config.training), "runtime": asdict(config.runtime),
        "monitoring": {"log_every": config.wandb.log_every,
                       "system_sample_seconds": config.wandb.system_sample_seconds},
        "data": {name: getattr(config.data, name) for name in
                 ("frames", "height", "width", "frame_stride", "batch_size", "horizontal_flip")},
        "teacher_sha256": provenance.get("teacher_sha256"),
        "student_sha256": provenance.get("student_sha256"),
        "source_commit": provenance.get("source", {}).get("commit"),
        "source_files": provenance.get("source", {}).get("files", {}),
        "manifests": provenance.get("data", {}),
        "training_plan": provenance.get("training_plan", {}),
    }


def run_labels(config, job_type, run_id):
    """Tags identify families; numeric hyperparameters remain structured config."""
    t, width = config.training, config.width
    recipe = "rec-gan" if t.reconstruction_updates and t.adversarial_updates else (
        "gan" if t.adversarial_updates else "rec")
    method = "original" if width.stages == ORIGINAL_WIDTHS and not width.hidden else "width-only"
    automatic = [f"model:{config.model.adapter}", f"method:{method}",
                 f"init:{config.model.init}", f"recipe:{recipe}", f"purpose:{job_type}"]
    if width.stages == [512, 512, 256, 64, 32] and not width.hidden:
        automatic.append("width:amd-v1-v3")
    # Automatic namespaces have one source of truth; user tags add dataset/campaign labels.
    reserved = {tag.split(":", 1)[0] for tag in automatic} | {"width"}
    extra = [tag for tag in config.wandb.tags if tag.split(":", 1)[0] not in reserved]
    tags = list(dict.fromkeys(automatic + extra))
    widths = "-".join(map(str, width.stages))
    name = config.wandb.name or (
        f"{config.model.adapter}-w{widths}-{config.model.init}-{recipe}-s{t.seed}-{run_id}")
    if job_type != "decoder-recovery" and not config.wandb.name:
        name = f"{job_type}-{name}"
    return name, tags


def empty_window():
    return {"count": 0, "sums": {}, "counts": {}, "maxima": {}, "last": {}}


class WandbTracker:
    def __init__(self, config, output, provenance=None, state=None, job_type="decoder-recovery"):
        self.config, self.output = config, Path(output)
        self.state = copy.deepcopy(state or {})
        self.run = None
        self.sdk_seconds = self.sdk_max_seconds = self.training_seconds = 0.0
        self.sdk_calls = 0
        self._log_failed = False
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
        name, tags = run_labels(config, job_type, run_id)
        directory = self.output / "wandb-logs"
        directory.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            entity=w.entity, project=w.project, id=run_id,
            name=None if resume_online else name,
            group=w.group or None, tags=tags, job_type=job_type,
            mode=w.mode, resume="must" if resume_online else ("never" if w.mode == "online" else None),
            dir=str(directory), config=public_config(config, provenance or {}),
            save_code=False, force=True, allow_val_change=resume_online,
            settings=wandb.Settings(console="off", disable_code=True, disable_git=True, init_timeout=30,
                                    x_stats_sampling_interval=float(w.system_sample_seconds)),
        )
        self.state.update(run_id=self.run.id, entity=w.entity, project=w.project, mode=w.mode)
        self.state.setdefault("window", empty_window())
        self.state.setdefault("best_quality", {})
        try:
            if previous_id and not resume_online:
                self.run.summary["resumed_from_offline_run"] = previous_id
            self.run.define_metric("progress/generator_updates")
            for pattern in ("train/*", "gan/*", "optim/*", "quality/*", "timing/*", "progress/*", "monitor/*"):
                self.run.define_metric(pattern, step_metric="progress/generator_updates")
            self.run.summary.update({"monitoring_schema": 2, "session_status": "running"})
            if job_type == "decoder-recovery":
                self.run.summary["training_complete"] = False
            write_json({key: self.state[key] for key in ("run_id", "entity", "project", "mode")} |
                       {"url": self.run.url}, self.output / "wandb-run.json")
        except BaseException:
            self.run.finish(exit_code=1)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.run is None:
            return False
        try:
            if not self._log_failed:
                self.flush_training(reason="interrupted" if exc_type else "close")
            self.run.summary.update({"session_status": "failed" if exc_type else "finished",
                                     "failure_type": exc_type.__name__ if exc_type else "",
                                     "monitor/sdk_log_calls": self.sdk_calls,
                                     "monitor/sdk_log_seconds": self.sdk_seconds,
                                     "monitor/sdk_log_ms_max": 1000 * self.sdk_max_seconds})
        except Exception:
            if exc_type is None:
                raise
        finally:
            self.run.finish(exit_code=1 if exc_type or self._log_failed else 0)
        return False

    def _log(self, data):
        started = time.perf_counter()
        try:
            self.run.log(data)
        except BaseException:
            self._log_failed = True
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.sdk_seconds += elapsed
            self.sdk_max_seconds = max(self.sdk_max_seconds, elapsed)
            self.sdk_calls += 1

    def log_training(self, metrics, trainer):
        if self.run is None:
            return
        update, stage = metrics["generator_updates"], metrics["stage"]
        window = self.state["window"]
        if window["count"] and update != window["context"]["progress/generator_updates"] + 1:
            # Tracking may have been disabled for part of a resumed training job.
            self.flush_training(reason="resume_gap")
            window = self.state["window"]
        if window["count"] and window["stage"] != stage:
            self.flush_training(reason="stage_end")
            window = self.state["window"]
        window["count"] += 1
        window.setdefault("first_update", update)
        window["stage"] = stage
        for key in SCALAR_GROUPS:
            if key in metrics:
                value = metrics[key]  # Trainer supplies CPU scalars, never CUDA tensors.
                window["sums"][key] = window["sums"].get(key, 0.0) + value
                window["counts"][key] = window["counts"].get(key, 0) + 1
                window["maxima"][key] = max(window["maxima"].get(key, value), value)
                window["last"][key] = value
        self.training_seconds += metrics["update_seconds"]
        t = self.config.training
        total = t.reconstruction_updates + t.adversarial_updates
        window["context"] = {
            "progress/generator_updates": update,
            "progress/discriminator_updates": metrics["discriminator_updates"],
            "progress/microbatches": metrics["microbatches"],
            "progress/fraction": update / total, "progress/epoch": trainer.stream.epoch,
            "progress/dataset_cursor": trainer.stream.cursor,
            "progress/stage": stage, "gan/active": int(stage == "adversarial"),
            "optim/generator_lr": trainer.optimizer.param_groups[0]["lr"],
        }
        if trainer.d_optimizer:
            window["context"]["optim/discriminator_lr"] = trainer.d_optimizer.param_groups[0]["lr"]
        # Always absolute optimizer updates: wall time NEVER decides a training record.
        if update == total or update == t.reconstruction_updates:
            self.flush_training(reason="complete" if update == total else "stage_end")
        elif update % self.config.wandb.log_every == 0:
            self.flush_training(reason="interval")
        elif update in {1, t.reconstruction_updates + 1}:
            # Preview first update without removing it from the next 50-step window.
            self.flush_training(reason="stage_start", reset=False)

    def flush_training(self, reason, reset=True):
        if self.run is None or not self.state["window"]["count"]:
            return
        w = self.state["window"]
        data = dict(w["context"])
        data.update({"monitor/window_updates": w["count"], "monitor/window_start": w["first_update"],
                     "monitor/record_reason": reason,
                     "monitor/sdk_log_calls_so_far": self.sdk_calls,
                     "monitor/sdk_log_ms_max_so_far": 1000 * self.sdk_max_seconds,
                     "monitor/sdk_log_seconds_so_far": self.sdk_seconds})
        for key, group in SCALAR_GROUPS.items():
            if key in w["sums"]:
                data[f"{group}/{key}_mean"] = w["sums"][key] / w["counts"][key]
                data[f"{group}/{key}_max"] = w["maxima"][key]
        # Last value accompanies averages, so a spike is not hidden by smoothing.
        data["train/loss_last"] = w["last"]["loss"]
        self._log(data)
        if reset:
            self.state["window"] = empty_window()

    def log_validation(self, result, updates, weights="ema", seconds=None):
        if self.run is None:
            return
        data = {"progress/generator_updates": updates, "quality/clips": result["clips"],
                "quality/weights": weights}
        if seconds is not None:
            data["timing/validation_seconds"] = seconds
        means = result["means"]
        summary = {}
        for label in ("teacher", "student"):
            for metric, value in means[label].items():
                data[f"quality/{label}_{metric}"] = value
        for metric in means["student"].keys() & means["teacher"].keys():
            data[f"quality/student_minus_teacher_{metric}"] = means["student"][metric] - means["teacher"][metric]
        for metric, value in means["student"].items():
            old = self.state["best_quality"].get(metric)
            improved = old is None or (value < old["value"] if metric == "lpips" else value > old["value"])
            if improved:
                self.state["best_quality"][metric] = {"value": value, "update": updates}
            best = self.state["best_quality"][metric]
            summary[f"best/{metric}"] = best["value"]
            summary[f"best/{metric}_update"] = best["update"]
        self._log(data)
        self.run.summary.update(summary)

    def log_benchmark(self, result):
        """Call only after timed decoding is over; SDK never runs inside its timers."""
        if self.run is None:
            return
        data = {}
        for label in ("native_teacher", "runtime_teacher", "runtime_student"):
            row = result[label]
            for name in ("median_ms", "p10_ms", "p90_ms", "speedup_vs_native", "time_reduction_percent",
                         "resident_allocated_bytes", "peak_additional_allocated_bytes"):
                if row.get(name) is not None:
                    data[f"decode/{label}_{name}"] = row[name]
        data["decode/student_over_runtime_teacher"] = result["student_over_runtime_teacher"]
        self._log(data)
        row = result["runtime_student"]
        self.run.summary.update({f"benchmark/{key}": row[key] for key in
                                 ("hardware", "torch", "cuda", "cudnn", "warmup", "repeats", "latent_shape", "scope")})

    def mark_complete(self, trainer):
        if self.run is not None:
            t = self.config.training
            complete = trainer.updates >= t.reconstruction_updates + t.adversarial_updates
            self.run.summary.update({"training_complete": complete,
                                     "session_stop_reason": "budget_reached" if complete else "review",
                                     "generator_updates": trainer.updates,
                                     "discriminator_updates": trainer.d_updates,
                                     "microbatches": trainer.microbatches})


def check_connection(entity, project, output):
    """Create one clearly labelled metrics-only check, without loading a model."""
    from .config import Config, WandbConfig
    config = Config(wandb=WandbConfig(enabled=True, entity=entity, project=project,
                                      name="connection-check", group="setup", tags=["connection-check"]))
    with WandbTracker(config, output, job_type="connection-check") as tracker:
        tracker._log({"setup/connection_ok": 1, "setup/optimizer_updates": 0})
        tracker.run.summary.update({"purpose": "W&B connection check only", "formal_training_started": False})
        result = {key: tracker.state[key] for key in ("run_id", "entity", "project", "mode")}
        result.update(url=tracker.run.url, connection_ok=True, formal_training_started=False)
    write_json(result, Path(output) / "connection-check.json")
    return result
