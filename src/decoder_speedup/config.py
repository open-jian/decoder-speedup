"""Strict, versioned configuration. Budgets always mean optimizer updates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import os
import re
import yaml

ORIGINAL_WIDTHS = [1024, 1024, 1024, 512, 256]
BLOCK_NAMES = ["middle.0", "middle.2"] + [
    f"upsamples.{stage}.upsamples.{block}" for stage in range(4) for block in range(3)
]


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class WidthConfig:
    # [entry/middle, up0, up1, up2, final stage]
    stages: list[int] = field(default_factory=lambda: ORIGINAL_WIDTHS.copy())
    hidden: dict[str, int] = field(default_factory=dict)

    def validate(self):
        if len(self.stages) != 5:
            raise ValueError("width.stages needs 5 widths: middle, up0, up1, up2, final")
        for i, (width, maximum) in enumerate(zip(self.stages, ORIGINAL_WIDTHS)):
            positive_int(width, f"width.stages[{i}]")
            if width > maximum:
                raise ValueError(f"Stage {i} exceeds original width {maximum}")
        for i, factor in enumerate((8, 8, 4)):
            if self.stages[i + 1] * factor % self.stages[i]:
                raise ValueError(f"Wan DupUp3D stage {i}: output width * {factor} must divide by input width")
        for name, width in self.hidden.items():
            if name not in BLOCK_NAMES:
                raise ValueError(f"Unknown residual block: {name}")
            positive_int(width, name)
            maximum = 1024 if name.startswith("middle") else ORIGINAL_WIDTHS[int(name.split('.')[1]) + 1]
            if width > maximum:
                raise ValueError(f"{name} hidden width exceeds original {maximum}")
        return self


@dataclass
class ModelConfig:
    adapter: str = "wan22"
    source: str = ""
    weights: str = ""
    init: str = "teacher_prefix"  # random | teacher_prefix | checkpoint
    student_checkpoint: str = ""


@dataclass
class DataConfig:
    root: str = ""
    train_manifest: str = ""
    val_manifest: str = ""
    frames: int = 17
    height: int = 256
    width: int = 256
    frame_stride: int = 1
    batch_size: int = 1
    horizontal_flip: bool = True


@dataclass
class LossConfig:
    l1: float = 1.0
    lpips: float = 1.0
    feature: float = 1.0
    features: list[str] = field(default_factory=lambda: ["middle", "upsamples.0"])
    gan: float = 0.05
    adaptive_gan: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    # Example budgets, NOT the unpublished authors' final recipe.
    reconstruction_updates: int = 20000
    adversarial_updates: int = 0
    accumulation: int = 1
    discriminator_updates: int = 1
    learning_rate: float = 1e-4
    discriminator_lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 1e-4
    epsilon: float = 1e-8
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    save_every: int = 1000
    eval_every: int = 1000
    eval_clips: int = 16
    discriminator_channels: int = 32


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    precision: str = "bf16"  # AMP; master weights stay FP32 during training
    weight_dtype: str = "fp32"  # bf16 allowed for inference, never for optimizer master weights
    channels_last: bool = False
    compile: bool = False  # inference only; not part of training/checkpoint state


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: str = ""
    project: str = "decoder-speedup"
    mode: str = "online"
    name: str = ""
    group: str = "wan22-width"
    tags: list[str] = field(default_factory=list)
    log_every: int = 10


@dataclass
class Config:
    version: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    width: WidthConfig = field(default_factory=WidthConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output: str = "runs/wan22"

    def validate(self):
        if self.version != 1 or self.model.adapter != "wan22":
            raise ValueError("This version supports version=1 and adapter=wan22")
        self.width.validate()
        w = self.wandb
        if type(w.enabled) is not bool or w.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb requires boolean enabled and mode online/offline/disabled")
        positive_int(w.log_every, "wandb.log_every")
        if w.enabled and (not w.entity or not w.project or "/" in w.entity or "/" in w.project):
            raise ValueError("Set wandb.entity and wandb.project separately")
        if not isinstance(w.tags, list) or not all(isinstance(tag, str) for tag in w.tags):
            raise ValueError("wandb.tags must be a list of strings")
        if self.model.init not in {"random", "teacher_prefix", "checkpoint"}:
            raise ValueError("Unknown initialization")
        if self.model.init == "checkpoint" and not self.model.student_checkpoint:
            raise ValueError("checkpoint initialization requires student_checkpoint")
        if self.runtime.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.runtime.weight_dtype not in {"fp32", "bf16"}:
            raise ValueError("weight_dtype must be fp32 or bf16")
        for name in ("frames", "height", "width", "frame_stride", "batch_size"):
            positive_int(getattr(self.data, name), f"data.{name}")
        if (self.data.frames - 1) % 4 or self.data.height % 16 or self.data.width % 16:
            raise ValueError("Wan clips need T=1+4k and H,W divisible by 16")
        t = self.training
        for name in ("reconstruction_updates", "adversarial_updates"):
            if type(getattr(t, name)) is not int or getattr(t, name) < 0:
                raise ValueError(f"training.{name} must be a nonnegative integer")
        if t.reconstruction_updates + t.adversarial_updates < 1:
            raise ValueError("At least one optimizer update is required")
        for name in ("accumulation", "discriminator_updates", "save_every", "eval_every", "eval_clips", "discriminator_channels"):
            positive_int(getattr(t, name), f"training.{name}")
        for name in ("learning_rate", "discriminator_lr", "epsilon"):
            if getattr(t, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if not 0 <= t.ema_decay < 1 or t.weight_decay < 0 or t.gradient_clip < 0:
            raise ValueError("Invalid EMA / weight decay / gradient clip")
        if len(t.betas) != 2 or any(not 0 <= b < 1 for b in t.betas):
            raise ValueError("Adam betas need two values in [0,1)")
        if any(getattr(self.loss, name) < 0 for name in ("l1", "lpips", "feature", "gan")):
            raise ValueError("Loss weights cannot be negative")
        if self.loss.l1 + self.loss.lpips + self.loss.feature <= 0:
            raise ValueError("At least one reconstruction/feature loss is required")
        if t.adversarial_updates and self.loss.adaptive_gan and self.loss.l1 + self.loss.lpips <= 0:
            raise ValueError("Adaptive GAN requires L1 or LPIPS reconstruction loss")
        if t.adversarial_updates and self.loss.gan <= 0:
            raise ValueError("Adversarial stage requires positive loss.gan")
        allowed = {"middle", *(f"upsamples.{i}" for i in range(4))}
        if not set(self.loss.features) <= allowed or len(set(self.loss.features)) != len(self.loss.features):
            raise ValueError("Invalid or duplicate feature paths")
        if self.loss.feature and not self.loss.features:
            raise ValueError("Feature loss requires at least one feature path")
        return self

    def to_dict(self):
        return asdict(self)


def _construct(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = value.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**value)


def from_dict(raw):
    raw = dict(raw)
    for key, cls in {"model": ModelConfig, "width": WidthConfig, "data": DataConfig,
                     "loss": LossConfig, "training": TrainConfig, "runtime": RuntimeConfig, "wandb": WandbConfig}.items():
        if key in raw:
            raw[key] = _construct(cls, raw[key])
    return _construct(Config, raw).validate()


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for k, v in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(v, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path):
    # Paths are relative to the config file, never the caller's current directory.
    path = Path(path).resolve()
    raw = yaml.load(path.read_text(), Loader=UniqueLoader)
    cfg = from_dict(raw)
    for obj, keys in [(cfg.model, ("source", "weights", "student_checkpoint")),
                      (cfg.data, ("root", "train_manifest", "val_manifest")), (cfg, ("output",))]:
        for key in keys:
            value = getattr(obj, key)
            if not value:
                continue
            value = os.path.expandvars(os.path.expanduser(value))
            if re.search(r"\$\{?\w+", value):
                raise ValueError(f"Unresolved environment variable in {key}: {value}")
            p = Path(value)
            setattr(obj, key, str(p.resolve() if p.is_absolute() else (path.parent / p).resolve()))
    return cfg
