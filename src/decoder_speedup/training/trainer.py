"""Single-GPU recovery training with explicit G/D updates and exact batch resume."""
from __future__ import annotations

from contextlib import contextmanager
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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": {str(device): torch.cuda.get_rng_state(device)} if device.type == "cuda" else {}}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    for device, value in state["cuda"].items():
        torch.cuda.set_rng_state(value.cpu(), device)


class Trainer:
    def __init__(self, student, teacher, stream, config, provenance=None, perceptual=None):
        config.validate()
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
        t = config.training
        kwargs = dict(betas=tuple(t.betas), weight_decay=t.weight_decay, eps=t.epsilon)
        self.parameters = [p for p in self.student.parameters() if p.requires_grad] + list(self.alignment.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=t.learning_rate, **kwargs)
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=t.discriminator_lr, **kwargs) if self.discriminator else None
        # EMA includes decoder and frozen normalization/latent projection, not teacher/D/feature heads.
        self.ema = {k: v.detach().clone() for k, v in self.student.state_dict().items()}
        self.updates = self.d_updates = self.microbatches = 0
        self.tracking_state = {}

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
        for _ in range(t.accumulation):
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
                    (loss_d / len(d_batches)).backward()
                    d_metric = d_metric + loss_d.detach() / len(d_batches)
                d_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), t.gradient_clip if t.gradient_clip else float("inf"), error_if_nonfinite=True)
                self.d_optimizer.step()
                self.d_updates += 1
                metrics["d_loss"] = metrics.get("d_loss", 0.0) + d_metric / t.discriminator_updates
                metrics["d_grad_norm"] = metrics.get("d_grad_norm", 0.0) + d_norm.detach() / t.discriminator_updates
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
        atomic_torch_save({"format": TRAINING_FORMAT, "config": self.config.to_dict(),
                           "provenance": self.provenance, "tracking": self.tracking_state, "student": self.student.state_dict(),
                           "alignment": self.alignment.state_dict(), "optimizer": self.optimizer.state_dict(),
                           "discriminator": self.discriminator.state_dict() if self.discriminator else None,
                           "d_optimizer": self.d_optimizer.state_dict() if self.d_optimizer else None,
                           "ema": self.ema, "updates": self.updates, "d_updates": self.d_updates,
                           "microbatches": self.microbatches, "stream": self.stream.state_dict(), "rng": rng_state(self.device)}, path)

    def resume(self, path):
        # Training checkpoints contain Python/NumPy RNG state. Only load your own trusted files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") not in TRAINING_FORMATS:
            raise ValueError("Not a training checkpoint")
        previous, current = copy.deepcopy(state["config"]), self.config.to_dict()
        # A relocated output directory is harmless. Hyperparameters/budgets stay strict.
        previous.pop("output", None)
        current.pop("output", None)
        # Logging frequency/name/enablement does not change optimizer or data semantics.
        previous.pop("wandb", None)
        current.pop("wandb", None)
        if previous != current or state["provenance"] != self.provenance:
            raise ValueError("Resume requires matching config and model/data provenance; use init=checkpoint for a new recipe")
        self.student.load_state_dict(state["student"])
        self.alignment.load_state_dict(state["alignment"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.discriminator:
            self.discriminator.load_state_dict(state["discriminator"])
            self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.ema = {k: v.to(self.device) for k, v in state["ema"].items()}
        self.updates, self.d_updates, self.microbatches = state["updates"], state["d_updates"], state["microbatches"]
        self.stream.load_state_dict(state["stream"])
        restore_rng(state["rng"])
        self.tracking_state = dict(state.get("tracking", {}))

    def fit(self, output, evaluate=None, tracker=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        t = self.config.training
        last_evaluation = None
        with (output / "train.jsonl").open("a", buffering=1) as log:
            while self.updates < t.reconstruction_updates + t.adversarial_updates:
                started = time.perf_counter()
                metrics = self.step()
                metrics["update_seconds"] = time.perf_counter() - started
                log.write(json.dumps(metrics) + "\n")
                if self.updates % self.config.wandb.log_every == 0 or self.updates in {
                    1, t.reconstruction_updates, t.reconstruction_updates + 1,
                    t.reconstruction_updates + t.adversarial_updates,
                }:
                    print(json.dumps(metrics), flush=True)
                if tracker is not None:
                    tracker.log_training(metrics, self)
                if evaluate and self.updates % t.eval_every == 0:
                    validation_started = time.perf_counter()
                    with self.ema_weights() as model:
                        result = evaluate(model)
                    last_evaluation = (self.updates, result)
                    (output / f"eval-{self.updates:08d}.json").write_text(json.dumps(result, indent=2) + "\n")
                    if tracker is not None:
                        tracker.log_validation(result, self.updates, seconds=time.perf_counter() - validation_started)
                if self.updates % t.save_every == 0 or self.updates == t.reconstruction_updates:
                    self.save(output / f"checkpoint-{self.updates:08d}.pt")
            self.save(output / "last.pt")
        return last_evaluation[1] if last_evaluation and last_evaluation[0] == self.updates else None
