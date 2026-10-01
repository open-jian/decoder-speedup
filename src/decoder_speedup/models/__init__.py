"""Model-specific adapters; the trainer only consumes latent/target/features."""
from typing import Protocol
import torch


class TeacherAdapter(Protocol):
    def prepare(self, video: torch.Tensor, feature_paths: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return model input and detached teacher feature targets."""
        ...
