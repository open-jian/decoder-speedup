"""Reuse official Wan2.2 layers and forward logic without importing its DiT stack."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import torch
from torch import nn
from ...config import WidthConfig


def load_source(root):
    root = Path(root).resolve()
    path = root / "wan/modules/vae2_2.py"
    if not path.is_file():
        raise FileNotFoundError(f"Wan2.2 source not found: {path}")
    package = "_decoder_speedup_wan_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    name = package + ".vae2_2"
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(path.parent)]
    sys.modules[package] = pkg
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def build_decoder(source, width: WidthConfig, *, z_dim=48):
    width.validate()
    dims = width.stages
    # Same forward and primitive operators as upstream. Only channel sizes change.
    class ConfiguredDecoder(source.Decoder3d):
        def __init__(self):
            nn.Module.__init__(self)
            self.z_dim = z_dim
            self.num_res_blocks = 2
            self.temperal_upsample = [True, True, False]
            self.conv1 = source.CausalConv3d(z_dim, dims[0], 3, padding=1)
            self.middle = nn.Sequential(source.ResidualBlock(dims[0], dims[0]),
                                        source.AttentionBlock(dims[0]), source.ResidualBlock(dims[0], dims[0]))
            self.upsamples = nn.Sequential(*[
                source.Up_ResidualBlock(a, b, 0.0, 3, i < 2, i < 3)
                for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
            ])
            self.head = nn.Sequential(source.RMS_norm(dims[-1], images=False), nn.SiLU(),
                                      source.CausalConv3d(dims[-1], 12, 3, padding=1))
    decoder = ConfiguredDecoder()
    for name, hidden in width.hidden.items():
        block = decoder.get_submodule(name)
        block.residual[2] = source.CausalConv3d(block.in_dim, hidden, 3, padding=1)
        block.residual[3] = source.RMS_norm(hidden, images=False)
        block.residual[6] = source.CausalConv3d(hidden, block.out_dim, 3, padding=1)
    return decoder


@torch.no_grad()
def inherit_prefix(student, teacher):
    """A reproducible warm start, NOT an optimal pruning or function-preserving claim.

    Each axis takes leading channels. Q/K/V and temporal phases retain their groups.
    New residual projections (where width changes make an identity impossible) are
    initialized with a rectangular identity. DupUp3D still uses its native layout.
    """
    old = teacher.state_dict()
    copied, created = [], []
    for name, value in student.state_dict().items():
        if name not in old:
            if ".shortcut." not in name:
                raise ValueError(f"No teacher tensor for {name}")
            value.zero_()
            if name.endswith("weight"):
                n = min(value.shape[:2])
                index = torch.arange(n, device=value.device)
                value[index, index, 0, 0, 0] = 1
            created.append(name)
            continue
        previous = old[name]
        if previous.ndim != value.ndim or any(a > b for a, b in zip(value.shape, previous.shape)):
            raise ValueError(f"Cannot inherit {name}: {tuple(previous.shape)} -> {tuple(value.shape)}")
        groups = 3 if ".to_qkv." in name else 2 if ".time_conv." in name else 1
        if groups > 1:
            old_group, new_group = previous.shape[0] // groups, value.shape[0] // groups
            previous = torch.cat([previous[i * old_group:i * old_group + new_group] for i in range(groups)], 0)
        value.copy_(previous[tuple(slice(0, size) for size in value.shape)])
        copied.append(name)
    return {"strategy": "teacher_prefix", "copied_tensors": len(copied), "new_projection_tensors": created}


class Decoder(nn.Module):
    """Independent decoder: normalized Wan latent -> unclamped RGB in [-1,1] convention.

    Cache is local to each call and remains differentiable across chunks. No mutable
    cache is retained on this wrapper; repeated calls cannot leak video history.
    """
    def __init__(self, source, decoder, conv2, mean, inverse_std, width):
        super().__init__()
        self.source = source
        self.decoder = decoder
        self.conv2 = conv2.requires_grad_(False)
        self.width = width
        self.register_buffer("mean", mean.detach().clone().float())
        self.register_buffer("inverse_std", inverse_std.detach().clone().float())

    def forward(self, latent, feature_paths=()):
        if latent.ndim != 5 or latent.shape[1] != self.mean.numel() or min(latent.shape[2:]) < 1:
            raise ValueError("Expected nonempty B,C,T,H,W Wan latent")
        collected = {name: [] for name in feature_paths}
        handles = []
        for name in collected:
            handles.append(self.decoder.get_submodule("middle.2" if name == "middle" else name).register_forward_hook(
                lambda module, args, result, key=name: collected[key].append(result)))
        try:
            z = latent / self.inverse_std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1)
            z = self.conv2(z.to(self.conv2.weight.dtype))
            cache = [None] * self.source.count_conv3d(self.decoder)
            frames = [self.decoder(z[:, :, i:i + 1], feat_cache=cache, feat_idx=[0], first_chunk=i == 0)
                      for i in range(z.shape[2])]
            rgb = self.source.unpatchify(torch.cat(frames, dim=2), patch_size=2)
            features = {name: torch.cat(values, dim=2) for name, values in collected.items()}
            return rgb, features
        finally:
            for handle in handles:
                handle.remove()

    @property
    def last_weight(self):
        return self.decoder.head[-1].weight

    def feature_channels(self, name):
        return self.width.stages[0] if name == "middle" else self.width.stages[int(name.split('.')[1]) + 1]


def build_student(source, width, teacher=None, init="random"):
    decoder = build_decoder(source, width)
    if teacher is None:
        if init != "random":
            raise ValueError("Teacher required for non-random initialization")
        conv2 = source.CausalConv3d(48, 48, 1)
        mean, inv_std = torch.zeros(48), torch.ones(48)
    else:
        conv2 = copy.deepcopy(teacher.model.conv2)
        mean, inv_std = teacher.scale
    report = {"strategy": init}
    if init == "teacher_prefix":
        report = inherit_prefix(decoder, teacher.model.decoder)
    student = Decoder(source, decoder, conv2, mean, inv_std, width)
    return student, report


class WanTeacher:
    def __init__(self, source, weights, device="cpu"):
        self.source = source
        wrapper = source.Wan2_2_VAE(vae_pth=str(weights), dtype=torch.float32, device=device)
        self.model = wrapper.model.eval().requires_grad_(False)
        self.scale = wrapper.scale
        self.decoder = Decoder(source, self.model.decoder, self.model.conv2, *self.scale, WidthConfig()).eval()

    @torch.no_grad()
    def prepare(self, video, feature_paths):
        latent = self.model.encode(video, self.scale).detach()
        features = self.decoder(latent, feature_paths)[1] if feature_paths else {}
        return latent, features

    @torch.no_grad()
    def reconstruct(self, video):
        latent = self.model.encode(video, self.scale)
        return self.decoder(latent)[0]
