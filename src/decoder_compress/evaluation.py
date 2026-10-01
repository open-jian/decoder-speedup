"""RGB quality and synchronized decoder-only timing; no historical baseline reuse."""
from __future__ import annotations
from dataclasses import asdict
import math
import platform
import time
import torch
import torch.nn.functional as F
from .runtime import autocast, inference_decoder


def quality_metrics(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError("Quality metrics require matching RGB shapes")
    predicted = predicted.float().clamp(-1, 1).add(1).div(2)
    target = target.float().clamp(-1, 1).add(1).div(2)
    mse = (predicted - target).square().mean().item()
    # SSIM: Gaussian 11x11, sigma 1.5, valid window, data range 1; average BT/C/H/W.
    x = predicted.permute(0, 2, 1, 3, 4).flatten(0, 1)
    y = target.permute(0, 2, 1, 3, 4).flatten(0, 1)
    if min(x.shape[-2:]) < 11:
        raise ValueError("SSIM requires H,W >=11")
    coords = torch.arange(11, device=x.device).float() - 5
    g = torch.exp(-coords.square() / (2 * 1.5**2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, 11, 11)
    conv = lambda a: F.conv2d(a, kernel, groups=x.shape[1])
    ux, uy = conv(x), conv(y)
    vx, vy, covariance = conv(x * x) - ux * ux, conv(y * y) - uy * uy, conv(x * y) - ux * uy
    ssim = (((2 * ux * uy + 0.01**2) * (2 * covariance + 0.03**2)) /
            ((ux.square() + uy.square() + 0.01**2) * (vx + vy + 0.03**2))).mean().item()
    return {"psnr_db": -10 * math.log10(max(mse, 1e-12)), "ssim": ssim}


@torch.inference_mode()
def evaluate_dataset(student, teacher, dataset, runtime, count, perceptual=None):
    was_training = student.training
    student.eval()
    rows = []
    try:
        for index in range(min(count, len(dataset))):
            target = dataset.get(index, seed=0)[None].to(runtime.device)
            with autocast(runtime):
                latent, _ = teacher.prepare(target, [])
                reference = teacher.decoder(latent)[0]
                prediction = student(latent)[0]
            row = {"path": dataset.rows[index]["path"], "source_id": dataset.rows[index]["source_id"]}
            for label, value in (("teacher", reference), ("student", prediction)):
                row[label] = quality_metrics(value, target)
                if perceptual is not None:
                    row[label]["lpips"] = float(perceptual(value.float().clamp(-1, 1), target))
            rows.append(row)
    finally:
        student.train(was_training)
    keys = rows[0]["teacher"].keys() if rows else []
    averages = {label: {key: sum(r[label][key] for r in rows) / len(rows) for key in keys}
                for label in ("teacher", "student")}
    return {"clips": len(rows), "aggregation": "mean of per-clip metrics", "runtime": asdict(runtime),
            "manifest": dataset.identity, "means": averages, "per_clip": rows}


@torch.inference_mode()
def benchmark_decoder(model, latent, runtime, warmup=5, repeats=30):
    if warmup < 1 or repeats < 3:
        raise ValueError("Timing needs warmup>=1 and repeats>=3")
    device = torch.device(runtime.device)
    model = inference_decoder(model, runtime)
    latent = latent.to(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    with autocast(runtime):
        for _ in range(warmup):
            model(latent)
        sync()
        memory_base = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        milliseconds = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            output = model(latent)[0]
            sync()
            milliseconds.append(1000 * (time.perf_counter() - start))
            del output
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    values = torch.tensor(milliseconds, dtype=torch.float64)
    return {"median_ms": values.median().item(), "p10_ms": values.quantile(0.1).item(),
            "p90_ms": values.quantile(0.9).item(), "samples_ms": milliseconds,
            "peak_additional_allocated_bytes": peak - memory_base if peak is not None else None,
            "resident_allocated_bytes": memory_base, "latent_shape": list(latent.shape),
            "runtime": asdict(runtime), "warmup": warmup, "repeats": repeats,
            "scope": "normalized latent on device -> unclamped RGB; includes all temporal chunks, excludes encoder/H2D",
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "backend_flags": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "cudnn_deterministic": torch.backends.cudnn.deterministic,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}}
