import types
import pytest
import torch
from decoder_speedup.config import WidthConfig
from decoder_speedup.models.wan22.adapter import build_decoder, build_student, Decoder, inherit_prefix
from decoder_speedup.export import export_student, load_student


TINY_WIDTH = WidthConfig([8, 8, 8, 4, 2])


def tiny_teacher(source):
    decoder = source.Decoder3d(dim=2, z_dim=48, temperal_upsample=[True, True, False]).eval()
    conv2 = source.CausalConv3d(48, 48, 1)
    mean, inv = torch.randn(48) * 0.01, torch.ones(48) * 0.9
    model = types.SimpleNamespace(decoder=decoder, conv2=conv2, z_dim=48, _decoder_conv_overrides={})
    def clear_cache():
        model._feat_map = [None] * source.count_conv3d(decoder)
    model.clear_cache = clear_cache
    wrapped = Decoder(source, decoder, conv2, mean, inv, TINY_WIDTH).eval().requires_grad_(False)
    teacher = types.SimpleNamespace(model=model, scale=[mean, inv], decoder=wrapped)
    return teacher


def test_full_width_matches_official_and_causal_cache(source):
    torch.manual_seed(13)
    teacher = tiny_teacher(source)
    model, report = build_student(source, TINY_WIDTH, teacher, "teacher_prefix")
    model.eval()
    latent = torch.randn(1, 48, 3, 2, 2)
    with torch.no_grad():
        official = source.WanVAE_.decode(teacher.model, latent, teacher.scale)
        actual, features = model(latent, ["middle", "upsamples.0"])
        assert actual.shape == (1, 3, 9, 32, 32)
        torch.testing.assert_close(actual, official, rtol=0, atol=0)
        # conv2 processes the entire latent; different GEMM shapes may round differently.
        torch.testing.assert_close(model(latent[:, :, :2])[0], actual[:, :, :5], rtol=1e-5, atol=1e-5)
        changed = latent.clone()
        changed[:, :, 2] += 10
        torch.testing.assert_close(model(changed)[0][:, :, :5], actual[:, :, :5], rtol=0, atol=0)
        model(torch.randn_like(latent))
        torch.testing.assert_close(model(latent)[0], actual, rtol=0, atol=0)
        assert features["middle"].shape[2] == 3
        assert features["upsamples.0"].shape[2] == 5
    assert not report["new_projection_tensors"]


def test_nonuniform_and_internal_width_backward(source):
    width = WidthConfig([8, 4, 4, 2, 2], {"middle.0": 3, "upsamples.2.upsamples.1": 1})
    model, _ = build_student(source, width, tiny_teacher(source), "teacher_prefix")
    latent = torch.randn(1, 48, 3, 1, 1, requires_grad=True)
    rgb, _ = model(latent)
    rgb[:, :, -1:].square().mean().backward()
    assert rgb.shape == (1, 3, 9, 16, 16)
    assert model.decoder.middle[0].residual[2].out_channels == 3
    assert model.decoder.upsamples[2].upsamples[1].residual[6].in_channels == 1
    assert latent.grad[:, :, 0].abs().sum() > 0  # temporal cache keeps gradients across chunks
    assert model.last_weight.grad.isfinite().all()
    assert model.conv2.weight.grad is None
    assert sum(p.numel() for p in model.decoder.parameters()) < sum(p.numel() for p in build_decoder(source, TINY_WIDTH).parameters())


def test_grouped_initialization(source):
    teacher = tiny_teacher(source)
    with torch.no_grad():
        for name, param in teacher.model.decoder.named_parameters():
            if "to_qkv" in name or "time_conv" in name:
                param.copy_(torch.arange(param.shape[0]).view(-1, *([1] * (param.ndim - 1))).expand_as(param))
    model, _ = build_student(source, WidthConfig([4, 4, 4, 2, 1]), teacher, "teacher_prefix")
    assert model.decoder.middle[1].to_qkv.bias.tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
    assert model.decoder.upsamples[0].upsamples[3].time_conv.bias.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]


def test_export_roundtrip(source, source_root, tmp_path):
    model, _ = build_student(source, WidthConfig([8, 4, 4, 2, 2]), tiny_teacher(source), "teacher_prefix")
    model.eval()
    file = tmp_path / "student.pt"
    export_student(model, file, source_root)
    restored, artifact = load_student(file, source_root)
    latent = torch.randn(1, 48, 2, 1, 1)
    with torch.no_grad():
        torch.testing.assert_close(restored(latent)[0], model(latent)[0], rtol=0, atol=0)
    assert artifact["format"] == "decoder-speedup-student-v1"
    # Existing exported models remain readable after the package/directory rename.
    artifact["format"] = "decoder-compress-student-v1"
    torch.save(artifact, file)
    legacy, _ = load_student(file, source_root)
    with torch.no_grad():
        torch.testing.assert_close(legacy(latent)[0], model(latent)[0], rtol=0, atol=0)
    assert all(not any(word in key for word in ("encoder", "discriminator", "projection")) for key in artifact["state_dict"])
    artifact["source"]["files"]["vae2_2.py"] = "wrong"
    torch.save(artifact, file)
    with pytest.raises(ValueError, match="source files"):
        load_student(file, source_root)


def test_warm_start_rejects_different_latent_contract(source, source_root, tmp_path):
    from decoder_speedup.export import initialize_checkpoint
    teacher = tiny_teacher(source)
    model, _ = build_student(source, TINY_WIDTH, teacher, "teacher_prefix")
    other, _ = build_student(source, TINY_WIDTH, teacher, "random")
    file = tmp_path / "initial.pt"
    export_student(model, file, source_root)
    initialize_checkpoint(other, file, source_root)
    torch.testing.assert_close(other.last_weight, model.last_weight)
    with torch.no_grad():
        other.mean.add_(1)
    with pytest.raises(ValueError, match="latent contract"):
        initialize_checkpoint(other, file, source_root)


def test_amd_width_preset_preserves_wan_structure(source):
    import json
    from pathlib import Path
    import yaml
    from decoder_speedup.config import from_dict

    root = Path(__file__).resolve().parents[1]
    preset = yaml.safe_load((root / "configs/wan22/width.yaml").read_text())
    provenance = json.loads((root / "docs/amd_width_source.json").read_text())
    width = from_dict({"width": preset["width"]}).width
    for published in provenance["sources"]:
        blocks = published["decoder_block_out_channels"][::-1]
        assert width.stages == [blocks[0], *blocks]
    assert width.hidden == {}
    with torch.device("meta"):
        model = build_decoder(source, width)
    assert sum(isinstance(m, source.ResidualBlock) for m in model.modules()) == 14
    assert isinstance(model.middle[1], source.AttentionBlock)
    assert model.conv1.in_channels == 48 and model.conv1.out_channels == 512
    assert model.head[-1].in_channels == 32 and model.head[-1].out_channels == 12
    assert [group.avg_shortcut.repeats for group in model.upsamples[:3]] == [8, 4, 1]
    assert [group.upsamples[-1].mode for group in model.upsamples[:3]] == ["upsample3d", "upsample3d", "upsample2d"]
    assert [len(group.upsamples) for group in model.upsamples] == [4, 4, 4, 3]
    for stage, group in enumerate(model.upsamples):
        output = width.stages[stage + 1]
        for block in group.upsamples[:3]:
            assert block.residual[2].out_channels == output
            assert block.residual[6].in_channels == output
            assert block.residual[2].kernel_size == block.residual[6].kernel_size == (3, 3, 3)
            assert block.residual[2].groups == block.residual[6].groups == 1
    assert model.upsamples[1].upsamples[0].shortcut.kernel_size == (1, 1, 1)
