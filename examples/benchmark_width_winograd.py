"""Compare one exported student's native and accelerated decoding, without W&B.

Run on an idle NVIDIA GPU. These are synthetic normalized latents, so output
differences describe numerical agreement, not reconstruction quality.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

from decoder_speedup.export import load_student
from decoder_speedup.provenance import sha256, source_identity, write_json


def install_options(text):
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("--install-kwargs must contain a JSON object")
    if result.get("conv_channel_pairs") is not None:
        result["conv_channel_pairs"] = {tuple(pair) for pair in result["conv_channel_pairs"]}
    if "f43_workspace_mib" in result:
        result["f43_workspace_mib"] = {
            tuple(map(int, pair.split(","))): budget
            for pair, budget in result["f43_workspace_mib"].items()
        }
    return result


def numerical_error(actual, expected):
    delta = actual.float() - expected.float()
    return {"max_abs": delta.abs().max().item(),
            "mean_abs": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()}


@torch.inference_mode()
def decode(model, latent):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(latent)[0]


def measure(model, latent, warmup, repeats):
    for _ in range(warmup):
        decode(model, latent)
    torch.cuda.synchronize(latent.device)
    initial_bytes = torch.cuda.memory_allocated(latent.device)
    torch.cuda.reset_peak_memory_stats(latent.device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(latent.device)
        begin = time.perf_counter()
        output = decode(model, latent)
        torch.cuda.synchronize(latent.device)
        samples.append((time.perf_counter() - begin) * 1000)
        del output
    peak_bytes = torch.cuda.max_memory_allocated(latent.device)
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "samples_ms": samples, "allocated_before_bytes": initial_bytes,
            "peak_allocated_bytes": peak_bytes,
            "additional_peak_bytes": peak_bytes - initial_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, nargs="+", default=[9, 81])
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3, help="Timed calls per block")
    parser.add_argument("--rounds", type=int, default=3, help="One block per mode each round; order shuffled deterministically")
    parser.add_argument("--seed", type=int, default=827)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--install-kwargs", default="{}", help="JSON installer options; channel pairs are lists of two integers")
    args = parser.parse_args()
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error("height and width must be positive multiples of 16")
    if any(frames < 1 or (frames - 1) % 4 for frames in args.frames):
        parser.error("each frame count must be 1 + 4*k")
    if args.repeats < 1 or args.warmup < 1 or args.rounds < 1:
        parser.error("warmup, repeats, and rounds must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    from decoder_speedup.winograd import install
    model, artifact = load_student(args.student, args.source, device)
    state_layout = {name: (tuple(value.shape), value.data_ptr())
                    for name, value in model.state_dict().items()}
    handle = install(model, **install_options(args.install_kwargs))
    rows = []
    start = time.perf_counter()
    try:
        for frames in args.frames:
            latent = torch.randn(1, 48, 1 + (frames - 1) // 4,
                                 args.height // 16, args.width // 16, device=device)
            blocks = {"native_student": [], "winograd_student": []}
            block_order = []
            rng = random.Random(args.seed + frames)
            for _ in range(args.rounds):
                order = list(blocks)
                rng.shuffle(order)
                for mode in order:
                    handle.disable() if mode == "native_student" else handle.enable()
                    blocks[mode].append(measure(model, latent, args.warmup, args.repeats))
                    block_order.append(mode)
            def aggregate(values):
                samples = [sample for block in values for sample in block["samples_ms"]]
                return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
                        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                        "samples_ms": samples, "blocks": values}
            native = aggregate(blocks["native_student"])
            accelerated = aggregate(blocks["winograd_student"])
            handle.disable()
            expected = decode(model, latent)
            handle.enable()
            actual = decode(model, latent)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise AssertionError("accelerated decoding returned an invalid output")
            error = numerical_error(actual, expected)
            handle.disable()
            restored = decode(model, latent)
            restored_exact = torch.equal(restored, expected)
            if not restored_exact:
                raise AssertionError("disabling acceleration did not restore the native output")
            rows.append({"output_shape": list(actual.shape), "latent_shape": list(latent.shape),
                         "block_order": block_order,
                         "native_student": native, "winograd_student": accelerated,
                         "speedup_factor": native["median_ms"] / accelerated["median_ms"],
                         "latency_reduction_percent": 100 * (1 - accelerated["median_ms"] / native["median_ms"]),
                         "error_vs_native_student": error, "disabled_output_bit_exact": restored_exact})
            del expected, actual, restored, latent
        if state_layout != {name: (tuple(value.shape), value.data_ptr())
                            for name, value in model.state_dict().items()}:
            raise AssertionError("the adapter changed checkpoint keys, shapes, or storage")
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        result = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "Same width-compressed checkpoint; full decoder only; no encoding, I/O, training, or W&B logging",
                  "precision": "FP32 stored weights, BF16 autocast; TF32 disabled",
                  "latent_contract": "Fixed-seed random normalized Wan latents; not a reconstruction-quality evaluation",
                  "measurement": "Synchronized host wall time after warmup; deterministically shuffled mode blocks; first-use JIT/weight transform excluded; memory includes shared cached allocations, not isolated-process footprints",
                  "device": str(device), "gpu": torch.cuda.get_device_name(device),
                  "compute_capability": list(torch.cuda.get_device_capability(device)),
                  "torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "project_revision": revision,
                  "source": source_identity(args.source), "student_path": str(args.student.resolve()),
                  "student_sha256": sha256(args.student), "width": artifact["width"],
                  "warmup_per_block": args.warmup, "repeats_per_block": args.repeats,
                  "rounds": args.rounds, "seed": args.seed,
                  "install_kwargs": json.loads(args.install_kwargs), "adapter_summary": handle.summary,
                  "implementation_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                                             for path in (Path(__file__).resolve().parents[1] / "src").rglob("*.py")},
                  "state_keys_shapes_storage_preserved": True,
                  "elapsed_seconds": time.perf_counter() - start, "rows": rows}
        write_json(result, args.output)
        print(json.dumps({"output": str(args.output.resolve()), "rows": rows}, indent=2))
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
