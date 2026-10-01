from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class FeatureAlignment(nn.Module):
    def __init__(self, student, teacher, paths):
        super().__init__()
        self.paths = list(paths)
        self.projections = nn.ModuleList([
            nn.Conv3d(student.feature_channels(p), teacher.feature_channels(p), 1)
            if student.feature_channels(p) != teacher.feature_channels(p) else nn.Identity()
            for p in paths
        ])

    def forward(self, student_features, teacher_features):
        losses = []
        for path, projection in zip(self.paths, self.projections):
            predicted, target = projection(student_features[path]), teacher_features[path]
            if predicted.shape != target.shape:
                raise ValueError(f"Feature shape mismatch at {path}: {predicted.shape} vs {target.shape}")
            losses.append(F.mse_loss(predicted.float(), target.detach().float()))
        return sum(losses)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS enabled: install decoder-speedup[perceptual] in your project environment") from exc
        self.net = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(self, predicted, target):
        # Framewise mean, with a bounded per-call activation footprint. Outputs are
        # not clamped during training so saturated predictions still receive gradients.
        p = predicted.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        t = target.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        with torch.autocast(p.device.type, enabled=False):
            return torch.stack([self.net(a[None], b[None]).mean() for a, b in zip(p, t)]).mean()


class PatchDiscriminator(nn.Module):
    """Small 3D PatchGAN; our Wan implementation, not a claimed Turbo checkpoint."""
    def __init__(self, channels=32):
        super().__init__()
        layers = []
        previous = 3
        for output in (channels, 2 * channels, 4 * channels):
            layers.extend([nn.Conv3d(previous, output, 3, stride=(1, 2, 2), padding=1), nn.LeakyReLU(0.2)])
            previous = output
        layers.append(nn.Conv3d(previous, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, video):
        return self.net(video)


def hinge_discriminator(real, fake):
    return 0.5 * (F.relu(1 - real.float()).mean() + F.relu(1 + fake.float()).mean())


def adaptive_weight(reconstruction_loss, adversarial_loss, last_weight):
    reconstruction_grad = torch.autograd.grad(reconstruction_loss, last_weight, retain_graph=True)[0]
    adversarial_grad = torch.autograd.grad(adversarial_loss, last_weight, retain_graph=True)[0]
    return (reconstruction_grad.float().norm() / (adversarial_grad.float().norm() + 1e-4)).clamp(0, 1e4).detach()
