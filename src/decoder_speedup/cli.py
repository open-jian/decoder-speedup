from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import torch
import yaml
from .config import load_config, from_dict, RuntimeConfig
from .formats import TRAINING_FORMATS
from .data import VideoDataset, BatchStream, assert_disjoint, make_manifests
from .models.wan22.adapter import load_source, build_student, build_decoder, WanTeacher
from .provenance import sha256, source_identity, write_json, framework_identity
from .export import export_student, load_student, initialize_checkpoint
from .runtime import apply_layout, inference_decoder
from .training.trainer import Trainer, seed_all
from .training.losses import PerceptualLoss
from .evaluation import evaluate_dataset, benchmark_decoder
from .tracking import WandbTracker, check_connection


def require_single_process():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This initial trainer is single-process; do not launch with multi-process torchrun")


def setup(cfg):
    source = load_source(cfg.model.source)
    teacher = WanTeacher(source, cfg.model.weights, "cpu")
    student, report = build_student(source, cfg.width, teacher,
                                    "teacher_prefix" if cfg.model.init == "teacher_prefix" else "random")
    if cfg.model.init == "checkpoint":
        report = initialize_checkpoint(student, cfg.model.student_checkpoint, cfg.model.source)
    teacher.model.to(cfg.runtime.device)
    teacher.scale = [v.to(cfg.runtime.device) for v in teacher.scale]
    teacher.decoder.to(cfg.runtime.device)
    apply_layout(teacher.model, cfg.runtime.channels_last)
    provenance = {"source": source_identity(cfg.model.source), "teacher_sha256": sha256(cfg.model.weights),
                  "initialization": report, "framework_version": "0.1.0", "framework_files": framework_identity(), "torch": str(torch.__version__)}
    return teacher, student, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(prog="decoder-speedup")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate structure and count parameters without loading weights")
    inspect.add_argument("config")
    manifest = commands.add_parser("manifest", help="Freeze source-disjoint train/validation JSONL")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--input-manifest")
    manifest.add_argument("--limit", type=int)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--validation-fraction", type=float, default=0.05)
    train = commands.add_parser("train")
    train.add_argument("config")
    train.add_argument("--resume", help="Trusted full training checkpoint; config must match")
    export = commands.add_parser("export")
    export.add_argument("checkpoint", help="Trusted training checkpoint")
    export.add_argument("--source", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--raw", action="store_true", help="Export current weights instead of EMA")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("config")
    evaluate.add_argument("--student", required=True)
    evaluate.add_argument("--output", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("config")
    benchmark.add_argument("--student", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--repeats", type=int, default=30)
    check = commands.add_parser("wandb-check", help="Create a connection-check run; no model or training")
    check.add_argument("--entity", default="miaoyin-uta")
    check.add_argument("--project", default="vae-speedup")
    check.add_argument("--output", default="runs/wandb-check")
    args = parser.parse_args(argv)
    if args.command == "wandb-check":
        print(json.dumps(check_connection(args.entity, args.project, args.output)))
        return
    if args.command == "manifest":
        train_rows, val_rows = make_manifests(args.root, args.output, input_manifest=args.input_manifest,
                                            limit=args.limit, seed=args.seed, validation_fraction=args.validation_fraction)
        print(json.dumps({"train": len(train_rows), "val": len(val_rows), "directory": str(Path(args.output).resolve())}))
        return
    if args.command == "export":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("format") not in TRAINING_FORMATS:
            raise ValueError("Not a training checkpoint")
        cfg = from_dict(checkpoint["config"])
        if source_identity(args.source)["files"] != checkpoint["provenance"]["source"]["files"]:
            raise ValueError("Wan source does not match training checkpoint")
        student, _ = build_student(load_source(args.source), cfg.width)
        weights = checkpoint["student"] if args.raw else checkpoint["ema"]
        student.load_state_dict(weights)
        provenance = dict(checkpoint["provenance"], generator_updates=checkpoint["updates"],
                          discriminator_updates=checkpoint["d_updates"], weights="raw" if args.raw else "ema")
        export_student(student, args.output, args.source, provenance)
        print(str(Path(args.output).resolve()))
        return
    cfg = load_config(args.config)
    if args.command == "inspect":
        source = load_source(cfg.model.source)
        with torch.device("meta"):
            student = build_decoder(source, cfg.width)
            original = source.Decoder3d(dim=256, z_dim=48, temperal_upsample=[True, True, False])
        count = lambda model: sum(p.numel() for p in model.parameters())
        print(json.dumps({"width": cfg.to_dict()["width"], "residual_blocks": 14,
                          "original_decoder_parameters": count(original), "student_decoder_parameters": count(student),
                          "parameter_ratio": count(student) / count(original), "source": source_identity(cfg.model.source),
                          "note": "Parameter reduction is not a measured speedup or restored quality result"}, indent=2))
        return
    require_single_process()
    seed_all(cfg.training.seed)
    if args.command == "train":
        if cfg.runtime.weight_dtype != "fp32":
            raise ValueError("Training requires weight_dtype=fp32")
        if cfg.runtime.compile:
            raise ValueError("Compilation is inference-only in this version")
        output = Path(cfg.output)
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Output directory is not empty; use a new run directory or --resume")
        train_data = VideoDataset(cfg.data, cfg.data.train_manifest)
        val_data = VideoDataset(cfg.data, cfg.data.val_manifest, training=False)
        assert_disjoint(train_data, val_data)
        teacher, student, provenance = setup(cfg)
        provenance["data"] = {"train": train_data.identity, "val": val_data.identity}
        stream = BatchStream(train_data, cfg.data.batch_size, cfg.training.seed)
        trainer = Trainer(student, teacher, stream, cfg, provenance)
        if args.resume:
            trainer.resume(args.resume)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        write_json(provenance, output / "provenance.json")
        evaluate = lambda model: evaluate_dataset(model, teacher, val_data, cfg.runtime, cfg.training.eval_clips, trainer.perceptual)
        with WandbTracker(cfg, output, provenance, trainer.tracking_state) as tracker:
            trainer.tracking_state = tracker.state
            trainer.fit(output, evaluate, tracker)
            # Always validate at the end, even for budgets shorter than eval_every.
            with trainer.ema_weights() as model:
                result = evaluate(model)
                write_json(result, output / "evaluation-final.json")
                tracker.log_validation(result, trainer.updates)
                export_student(model, output / "student.pt", cfg.model.source,
                               dict(provenance, generator_updates=trainer.updates, discriminator_updates=trainer.d_updates, weights="ema"))
            tracker.mark_complete(trainer)
        return
    source = load_source(cfg.model.source)
    teacher = WanTeacher(source, cfg.model.weights, cfg.runtime.device)
    student, artifact = load_student(args.student, cfg.model.source, cfg.runtime.device)
    if artifact.get("provenance", {}).get("teacher_sha256") not in (None, sha256(cfg.model.weights)):
        raise ValueError("Student and teacher checkpoint identity mismatch")
    if args.command == "evaluate":
        if cfg.runtime.compile:
            raise ValueError("Quality evaluation uses eager inference; benchmark compilation separately")
        dataset = VideoDataset(cfg.data, cfg.data.val_manifest, training=False)
        student = inference_decoder(student, cfg.runtime)
        teacher.decoder = inference_decoder(teacher.decoder, cfg.runtime)
        perceptual = PerceptualLoss().to(cfg.runtime.device) if cfg.loss.lpips else None
        result = evaluate_dataset(student, teacher, dataset, cfg.runtime, cfg.training.eval_clips, perceptual)
    else:
        # Same normalized latent for all three; no encoding, I/O, or training in timers.
        latent = torch.randn(cfg.data.batch_size, 48, 1 + (cfg.data.frames - 1) // 4,
                             cfg.data.height // 16, cfg.data.width // 16, device=cfg.runtime.device)
        native = RuntimeConfig(device=cfg.runtime.device, precision=cfg.runtime.precision)
        result = {"native_teacher": benchmark_decoder(teacher.decoder, latent, native, args.warmup, args.repeats),
                  "runtime_teacher": benchmark_decoder(teacher.decoder, latent, cfg.runtime, args.warmup, args.repeats),
                  "runtime_student": benchmark_decoder(student, latent, cfg.runtime, args.warmup, args.repeats)}
        base = result["native_teacher"]["median_ms"]
        for key in ("runtime_teacher", "runtime_student"):
            elapsed = result[key]["median_ms"]
            result[key]["speedup_vs_native"] = base / elapsed
            result[key]["time_reduction_percent"] = 100 * (1 - elapsed / base)
        result["student_over_runtime_teacher"] = result["runtime_teacher"]["median_ms"] / result["runtime_student"]["median_ms"]
    result["student_provenance"] = artifact.get("provenance", {})
    result["source"] = source_identity(cfg.model.source)
    write_json(result, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve())}))
