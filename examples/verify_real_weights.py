"""Integration check on synthetic tensors, not recovery training or a speed study.

Run explicitly on a verified idle GPU. No dataset download or training loop.
"""
import argparse
import gc
import json
from pathlib import Path
import yaml
import torch
from decoder_compress.config import WidthConfig, from_dict
from decoder_compress.models.wan22.adapter import load_source, WanTeacher, build_student
from decoder_compress.provenance import source_identity, sha256, write_json
from decoder_compress.export import export_student, load_student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    source = load_source(args.source)
    teacher = WanTeacher(source, args.weights, "cpu")
    same, _ = build_student(source, WidthConfig(), teacher, "teacher_prefix")
    teacher.model.to(args.device)
    teacher.scale = [v.to(args.device) for v in teacher.scale]
    teacher.decoder.to(args.device)
    same.to(args.device).eval()
    video = torch.randn(1, 3, 9, 32, 32, device=args.device).tanh()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent, features = teacher.prepare(video, ["middle", "upsamples.0"])
        official = teacher.model.decode(latent, teacher.scale)
        actual = same(latent)[0]
        difference = float((official.float() - actual.float()).abs().max())
        torch.testing.assert_close(actual, official, rtol=0, atol=0)
    del same
    gc.collect()
    torch.cuda.empty_cache()
    # Exercise the shipped width plan, not a separately invented example.
    preset = Path(__file__).resolve().parents[1] / "configs/wan22/width.yaml"
    width = from_dict({"width": yaml.safe_load(preset.read_text())["width"]}).width
    # Teacher remains frozen; one backward pass checks actual widths/cache/gradients.
    student, initialization = build_student(source, width, teacher, "teacher_prefix")
    student.to(args.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted, student_features = student(latent.clone(), ["middle", "upsamples.0"])
        loss = (predicted.float() - video).abs().mean()
    loss.backward()
    assert student.last_weight.grad is not None and student.last_weight.grad.isfinite().all()
    assert all(p.grad is None for p in teacher.decoder.parameters())
    student.zero_grad(set_to_none=True)
    artifact_path = args.output + ".student.pt"
    student.eval()
    export_student(student, artifact_path, args.source)
    restored, _ = load_student(artifact_path, args.source, args.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        one = student(latent)[0]
        two = restored(latent)[0]
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    result = {"scope": "synthetic 9-frame 32x32 integration; one backward, zero optimizer steps",
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "source": source_identity(args.source), "teacher_sha256": sha256(args.weights),
              "original_width_max_abs_error": difference,
              "compressed_width": width.stages, "compressed_hidden": width.hidden,
              "initialization": initialization, "rgb_shape": list(one.shape),
              "finite_student_gradient": True, "teacher_gradient_absent": True,
              "export_roundtrip_max_abs_error": float((one.float() - two.float()).abs().max()),
              "teacher_feature_shapes": {k: list(v.shape) for k,v in features.items()},
              "student_feature_shapes": {k: list(v.shape) for k,v in student_features.items()},
              "recovery_training_started": False, "quality_or_speed_claim": None}
    write_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
