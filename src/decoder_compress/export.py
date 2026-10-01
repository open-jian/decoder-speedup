"""Portable student artifacts: no teacher, discriminator or feature projections."""
from pathlib import Path
import torch
from .config import WidthConfig
from .models.wan22.adapter import load_source, build_student
from .provenance import atomic_torch_save, source_identity, sha256


def export_student(student, path, source, provenance=None, state_dict=None):
    state = student.state_dict() if state_dict is None else state_dict
    atomic_torch_save({"format": "decoder-compress-student-v1", "adapter": "wan22",
                       "width": {"stages": student.width.stages, "hidden": student.width.hidden},
                       "source": source_identity(source), "provenance": provenance or {},
                       "contract": {"latent_channels": 48, "latent_normalization": "Wan2.2 mean/inverse_std in state",
                                    "temporal_factor": 4, "spatial_factor": 16,
                                    "frames": "1+4*(latent_frames-1)", "output": "unclamped RGB, [-1,1] convention"},
                       "state_dict": {key: value.detach().cpu().clone() for key, value in state.items()}}, path)


def load_student(path, source, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") != "decoder-compress-student-v1" or artifact.get("adapter") != "wan22":
        raise ValueError("Expected a decoder-compress Wan2.2 student artifact")
    identity = source_identity(source)
    if identity["files"] != artifact["source"]["files"]:
        raise ValueError("Wan source files differ from exported artifact; use its recorded source version")
    model, _ = build_student(load_source(source), WidthConfig(**artifact["width"]))
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.to(device).eval()
    return model, artifact


def initialize_checkpoint(student, path, source):
    # A warm start is intentionally distinct from optimizer/RNG/data-stream resume.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") == "decoder-compress-student-v1":
        if state["source"]["files"] != source_identity(source)["files"]:
            raise ValueError("Initialization artifact source mismatch")
        width, weights = state["width"], state["state_dict"]
    elif state.get("format") == "decoder-compress-training-v1":
        if state["provenance"].get("source", {}).get("files") != source_identity(source)["files"]:
            raise ValueError("Initialization checkpoint source mismatch")
        width, weights = state["config"]["width"], state["ema"]
    else:
        raise ValueError("Unknown initialization checkpoint format")
    if width != {"stages": student.width.stages, "hidden": student.width.hidden}:
        raise ValueError("Student checkpoint width mismatch")
    # A decoder trained for another latent projection/normalization is not a valid
    # warm start for this fixed encoder, even when its tensor shapes match.
    current = student.state_dict()
    for name in ("conv2.weight", "conv2.bias", "mean", "inverse_std"):
        if not torch.equal(current[name].detach().cpu(), weights[name].detach().cpu()):
            raise ValueError(f"Student checkpoint has incompatible frozen latent contract: {name}")
    student.load_state_dict(weights)
    return {"strategy": "checkpoint", "path": str(Path(path).resolve()), "sha256": sha256(path), "weights": "EMA for training checkpoints"}
