"""Recovery training with globally counted G/D updates and resumable data parallelism."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..provenance import atomic_torch_save
from ..formats import TRAINING_FORMAT, TRAINING_FORMATS
from ..runtime import autocast, apply_layout
from .losses import FeatureAlignment, PerceptualLoss, PatchDiscriminator, hinge_discriminator, adaptive_weight
from .budget import resolve_budget
from ..config import from_dict, positive_int
from . import distributed as parallel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": {str(device): torch.cuda.get_rng_state(device)} if device.type == "cuda" else {}}


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    for saved_device, value in state["cuda"].items():
        torch.cuda.set_rng_state(value.cpu(), device if device is not None else saved_device)


class Trainer:
    def __init__(self, student, teacher, stream, config, provenance=None, perceptual=None):
        config.validate()
        self.training_plan = resolve_budget(config, len(stream.dataset), parallel.world_size())
        if config.runtime.weight_dtype != "fp32":
            raise ValueError("Training requires FP32 master weights; use precision=bf16 for AMP")
        if config.runtime.compile:
            raise ValueError("runtime.compile is inference-only; set false for training")
        self.config, self.student, self.teacher, self.stream = config, student, teacher, stream
        self.provenance = provenance or {}
        self.device = torch.device(config.runtime.device)
        self.student.to(self.device).train()
        apply_layout(self.student, config.runtime.channels_last)
        self.feature_paths = config.loss.features if config.loss.feature else []
        self.alignment = FeatureAlignment(student, teacher.decoder, self.feature_paths).to(self.device)
        self.perceptual = (perceptual if perceptual is not None else PerceptualLoss()).to(self.device) if config.loss.lpips else None
        self.discriminator = PatchDiscriminator(config.training.discriminator_channels).to(self.device) if config.training.adversarial_updates else None
        for module in (self.student, self.alignment, self.discriminator):
            parallel.broadcast_module(module)
        t = config.training
        kwargs = dict(betas=tuple(t.betas), weight_decay=t.weight_decay, eps=t.epsilon)
        self.parameters = [p for p in self.student.parameters() if p.requires_grad] + list(self.alignment.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=t.learning_rate, **kwargs)
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=t.discriminator_lr, **kwargs) if self.discriminator else None
        # EMA includes decoder and frozen normalization/latent projection, not teacher/D/feature heads.
        self.ema = {k: v.detach().clone() for k, v in self.student.state_dict().items()}
        self.updates = self.d_updates = self.microbatches = 0
        self.tracking_state = {}

    def _create_discriminator(self):
        t = self.config.training
        self.discriminator = PatchDiscriminator(t.discriminator_channels).to(self.device)
        parallel.broadcast_module(self.discriminator)
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=t.discriminator_lr,
                                           betas=tuple(t.betas), weight_decay=t.weight_decay, eps=t.epsilon)

    def _batch(self):
        video = self.stream.next().to(self.device)
        with torch.no_grad(), autocast(self.config.runtime):
            latent, features = self.teacher.prepare(video, self.feature_paths)
        return video, latent.detach(), {k: v.detach() for k, v in features.items()}

    def step(self):
        cfg, t = self.config, self.config.training
        total = t.reconstruction_updates + t.adversarial_updates
        if self.updates >= total:
            raise StopIteration("Configured optimizer update budget reached")
        gan_active = self.updates >= t.reconstruction_updates
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.discriminator:
            self.discriminator.requires_grad_(False)
        metrics, d_batches = {}, []
        for microbatch in range(t.accumulation):
            if microbatch % parallel.world_size() != parallel.rank():
                self.stream.next(decode=False)
                self.microbatches += 1
                continue
            target, latent, teacher_features = self._batch()
            with autocast(cfg.runtime):
                predicted, features = self.student(latent, self.feature_paths)
                if predicted.shape != target.shape:
                    raise ValueError(f"RGB shape mismatch: {predicted.shape} vs {target.shape}")
                l1 = F.l1_loss(predicted.float(), target.float())
                perceptual = self.perceptual(predicted, target) if self.perceptual else predicted.new_zeros(())
                feature = self.alignment(features, teacher_features) if self.feature_paths else predicted.new_zeros(())
                rec = cfg.loss.l1 * l1 + cfg.loss.lpips * perceptual
                loss = rec + cfg.loss.feature * feature
                g_loss, gan_weight = predicted.new_zeros(()), predicted.new_zeros(())
                if gan_active:
                    g_loss = -self.discriminator(predicted).float().mean()
                    gan_weight = cfg.loss.gan * (adaptive_weight(rec, g_loss, self.student.last_weight)
                                                if cfg.loss.adaptive_gan else 1.0)
                    loss = loss + gan_weight * g_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generator loss; no optimizer update applied")
            (loss / t.accumulation).backward()
            for key, value in {"loss": loss, "l1": l1, "lpips": perceptual, "feature": feature,
                               "g_gan": g_loss, "gan_weight": gan_weight}.items():
                # Keep detached scalars on device; transfer them together once per update.
                metrics[key] = metrics.get(key, 0.0) + torch.as_tensor(value, device=self.device).detach() / t.accumulation
            if gan_active:
                d_batches.append((target.detach(), predicted.detach()))
            self.microbatches += 1
        parallel.sum_gradients(self.parameters)
        metrics["g_grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters, t.gradient_clip if t.gradient_clip else float("inf"), error_if_nonfinite=True).detach()
        self.optimizer.step()
        self.updates += 1
        with torch.no_grad():
            for name, value in self.student.state_dict().items():
                if value.is_floating_point():
                    self.ema[name].lerp_(value.detach(), 1 - t.ema_decay)
                else:
                    self.ema[name].copy_(value)
        if gan_active:
            self.discriminator.requires_grad_(True).train()
            for _ in range(t.discriminator_updates):
                self.d_optimizer.zero_grad(set_to_none=True)
                d_metric = 0.0
                for real, fake in d_batches:
                    with autocast(cfg.runtime):
                        loss_d = hinge_discriminator(self.discriminator(real), self.discriminator(fake))
                    if not torch.isfinite(loss_d):
                        raise FloatingPointError("Nonfinite discriminator loss")
                    (loss_d / t.accumulation).backward()
                    d_metric = d_metric + loss_d.detach() / t.accumulation
                parallel.sum_gradients(self.discriminator.parameters())
                d_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), t.gradient_clip if t.gradient_clip else float("inf"), error_if_nonfinite=True)
                self.d_optimizer.step()
                self.d_updates += 1
                metrics["d_loss"] = metrics.get("d_loss", 0.0) + d_metric / t.discriminator_updates
                metrics["d_grad_norm"] = metrics.get("d_grad_norm", 0.0) + d_norm.detach() / t.discriminator_updates
        parallel.sum_metrics(metrics)
        values = torch.stack([value.float() for value in metrics.values()]).cpu().tolist()
        metrics = dict(zip(metrics, values))
        return {"generator_updates": self.updates, "discriminator_updates": self.d_updates,
                "microbatches": self.microbatches, "stage": "adversarial" if gan_active else "reconstruction", **metrics}

    @contextmanager
    def ema_weights(self):
        old = {k: v.detach().clone() for k, v in self.student.state_dict().items()}
        was_training = self.student.training
        self.student.load_state_dict(self.ema)
        self.student.eval()
        try:
            yield self.student
        finally:
            self.student.load_state_dict(old)
            self.student.train(was_training)

    def save(self, path):
        # Saved only at complete G/D update boundaries; gradients need not be serialized.
        states = parallel.gather_rng(rng_state(self.device))
        if not parallel.primary():
            return
        atomic_torch_save({"format": TRAINING_FORMAT, "config": self.config.to_dict(),
                           "training_plan": self.training_plan,
                           "provenance": self.provenance, "tracking": self.tracking_state, "student": self.student.state_dict(),
                           "alignment": self.alignment.state_dict(), "optimizer": self.optimizer.state_dict(),
                           "discriminator": self.discriminator.state_dict() if self.discriminator else None,
                           "d_optimizer": self.d_optimizer.state_dict() if self.d_optimizer else None,
                           "ema": self.ema, "updates": self.updates, "d_updates": self.d_updates,
                           "microbatches": self.microbatches, "stream": self.stream.state_dict(),
                           "rng": states[0], "rng_by_rank": states, "world_size": parallel.world_size()}, path)

    def resume(self, path, *, start_gan_updates=None, extend_reconstruction_updates=None, allow_framework_change=False):
        # Training checkpoints contain Python/NumPy RNG state. Only load your own trusted files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") not in TRAINING_FORMATS:
            raise ValueError("Not a training checkpoint")
        previous, current = from_dict(state["config"]).to_dict(), self.config.to_dict()
        old_budget = {key: previous["training"][key] for key in ("reconstruction_updates", "adversarial_updates")}
        if start_gan_updates is not None and extend_reconstruction_updates is not None:
            raise ValueError("Choose either GAN transition or reconstruction extension")
        changing_budget = start_gan_updates is not None or extend_reconstruction_updates is not None
        if changing_budget:
            value = start_gan_updates if start_gan_updates is not None else extend_reconstruction_updates
            positive_int(value, "additional optimizer updates")
            if state["d_updates"] or state["updates"] > old_budget["reconstruction_updates"]:
                raise ValueError("Stage changes require a reconstruction checkpoint, before any GAN updates")
            if state["updates"] == 0:
                raise ValueError("Stage continuation requires a checkpoint with completed reconstruction updates")
            for key in ("reconstruction_updates", "reconstruction_epochs", "adversarial_updates"):
                previous["training"].pop(key, None)
                current["training"].pop(key, None)
        # A relocated output directory is harmless. Hyperparameters/budgets stay strict.
        previous.pop("output", None)
        current.pop("output", None)
        # Logging frequency/name/enablement does not change optimizer or data semantics.
        previous.pop("wandb", None)
        current.pop("wandb", None)
        if parallel.active() and previous['runtime']['device'].startswith('cuda') and current['runtime']['device'].startswith('cuda'):
            previous['runtime']['device'] = current['runtime']['device'] = 'cuda:0'
        old_provenance, new_provenance = copy.deepcopy(state['provenance']), copy.deepcopy(self.provenance)
        framework_change = None
        if allow_framework_change:
            old_files = old_provenance.pop('framework_files', None)
            new_files = new_provenance.pop('framework_files', None)
            if old_files != new_files:
                framework_change = {'event': 'framework_change', 'previous_files': old_files, 'current_files': new_files,
                                    'at_generator_update': state['updates']}
        if previous != current or old_provenance != new_provenance:
            raise ValueError("Resume requires matching config and model/data provenance; use init=checkpoint for a new recipe")
        t = self.config.training
        if start_gan_updates is not None:
            t.reconstruction_updates, t.adversarial_updates = state["updates"], start_gan_updates
            t.reconstruction_epochs = None
        elif extend_reconstruction_updates is not None:
            if old_budget["adversarial_updates"]:
                raise ValueError("Cannot extend reconstruction when a GAN budget was already scheduled")
            t.reconstruction_updates = old_budget["reconstruction_updates"] + extend_reconstruction_updates
            t.adversarial_updates, t.reconstruction_epochs = 0, None
        self.config.validate()
        self.student.load_state_dict(state["student"])
        self.alignment.load_state_dict(state["alignment"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state["discriminator"] is not None:
            if self.discriminator is None:
                self._create_discriminator()
            self.discriminator.load_state_dict(state["discriminator"])
            self.d_optimizer.load_state_dict(state["d_optimizer"])
        else:
            self.discriminator = self.d_optimizer = None
        self.ema = {k: v.to(self.device) for k, v in state["ema"].items()}
        self.updates, self.d_updates, self.microbatches = state["updates"], state["d_updates"], state["microbatches"]
        self.stream.load_state_dict(state["stream"])
        states = state.get('rng_by_rank', [state['rng']])
        restore_rng(states[parallel.rank() % len(states)], self.device if self.device.type == 'cuda' else None)
        # A newly enabled D is initialized once, from the checkpoint RNG state.
        # Subsequent resumes restore its weights/optimizer and do not reinitialize it.
        if t.adversarial_updates and self.discriminator is None:
            self._create_discriminator()
        self.tracking_state = dict(state.get("tracking", {}))
        self.training_plan = copy.deepcopy(state.get("training_plan", self.training_plan))
        previous_world = state.get('world_size', 1)
        self.training_plan.update(world_size=parallel.world_size(), local_accumulation=t.accumulation // parallel.world_size())
        if framework_change:
            self.training_plan.setdefault('changes', []).append(framework_change)
        if previous_world != parallel.world_size():
            self.training_plan.setdefault('changes', []).append({'event': 'world_size_change',
                'at_generator_update': self.updates, 'previous_world_size': previous_world,
                'world_size': parallel.world_size(), 'effective_batch': self.config.data.batch_size * t.accumulation})
        if changing_budget:
            self.training_plan.setdefault("changes", []).append({
                "event": "start_gan" if start_gan_updates is not None else "extend_reconstruction",
                "at_generator_update": self.updates, "previous_budget": old_budget,
                "reconstruction_updates": t.reconstruction_updates, "adversarial_updates": t.adversarial_updates,
            })
            self.training_plan.update(reconstruction_updates=t.reconstruction_updates, adversarial_updates=t.adversarial_updates)

    def fit(self, output, evaluate=None, tracker=None, stop_after_updates=None):
        output = Path(output)
        if parallel.primary():
            output.mkdir(parents=True, exist_ok=True)
        parallel.barrier()
        t = self.config.training
        total = t.reconstruction_updates + t.adversarial_updates
        if stop_after_updates is not None:
            positive_int(stop_after_updates, "stop_after_updates")
            if stop_after_updates <= self.updates:
                raise ValueError("stop_after_updates must exceed the completed generator update count")
        stop_at = min(total, stop_after_updates) if stop_after_updates is not None else total
        last_evaluation = None
        with ((output / "train.jsonl").open("a", buffering=1) if parallel.primary() else nullcontext(None)) as log:
            while self.updates < stop_at:
                started = time.perf_counter()
                metrics = self.step()
                metrics["update_seconds"] = time.perf_counter() - started
                if log is not None:
                    log.write(json.dumps(metrics) + "\n")
                if parallel.primary() and (self.updates % self.config.wandb.log_every == 0 or self.updates in {
                    1, t.reconstruction_updates, t.reconstruction_updates + 1,
                    t.reconstruction_updates + t.adversarial_updates,
                }):
                    print(json.dumps(metrics), flush=True)
                if tracker is not None:
                    tracker.log_training(metrics, self)
                if evaluate and parallel.primary() and self.updates % t.eval_every == 0:
                    validation_started = time.perf_counter()
                    with self.ema_weights() as model:
                        result = evaluate(model)
                    last_evaluation = (self.updates, result)
                    (output / f"eval-{self.updates:08d}.json").write_text(json.dumps(result, indent=2) + "\n")
                    if tracker is not None:
                        tracker.log_validation(result, self.updates, seconds=time.perf_counter() - validation_started)
                parallel.barrier()
                if self.updates % t.save_every == 0 or self.updates == t.reconstruction_updates:
                    self.save(output / f"checkpoint-{self.updates:08d}.pt")
                if parallel.should_stop(output / 'stop.request', self.device):
                    break
            self.save(output / "last.pt")
        parallel.barrier()
        return last_evaluation[1] if last_evaluation and last_evaluation[0] == self.updates else None
