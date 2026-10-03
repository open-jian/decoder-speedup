"""Execution adapters must preserve the trained student's public contract."""

import pytest
import torch

from decoder_speedup.config import WidthConfig
from decoder_speedup.export import export_student, load_student
from decoder_speedup.models.wan22.adapter import build_student


def test_legacy_runtime_defaults_to_native():
    from decoder_speedup.config import RuntimeConfig, from_dict
    assert RuntimeConfig().winograd is False
    assert from_dict({"runtime": {"device": "cpu", "precision": "fp32"}}).runtime.winograd is False


@pytest.mark.parametrize("options", [
    {"winograd": "true"},
    {"winograd": True, "device": "cpu"},
    {"winograd": True, "precision": "fp32"},
    {"winograd": True, "weight_dtype": "bf16"},
    {"winograd": True, "compile": True},
])
def test_invalid_accelerated_runtime_is_rejected(options):
    from decoder_speedup.config import from_dict
    with pytest.raises(ValueError, match="winograd"):
        from_dict({"runtime": options})


def test_disabled_runtime_does_not_import_kernel_dependencies(monkeypatch):
    import builtins
    from decoder_speedup.config import RuntimeConfig
    from decoder_speedup.runtime import inference_decoder
    native_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if "winograd" in name or name == "triton" or name.startswith("triton."):
            raise AssertionError(f"native execution imported optional kernels: {name}")
        return native_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    model = torch.nn.Conv3d(2, 2, 1)
    actual = inference_decoder(model, RuntimeConfig(device="cpu", precision="fp32"))
    assert actual is model and not actual.training


@pytest.fixture
def student(source):
    pytest.importorskip("triton")
    torch.manual_seed(87)
    model, _ = build_student(source, WidthConfig([8, 8, 8, 4, 2]))
    return model.eval()


def installer():
    from decoder_speedup.winograd import install
    return install


def test_reversible_adapter_preserves_state_and_cpu_output(student):
    latent = torch.randn(1, 48, 3, 2, 3)
    before = {name: value.clone() for name, value in student.state_dict().items()}
    pointers = {name: value.data_ptr() for name, value in student.state_dict().items()}
    topology = [(name, type(layer)) for name, layer in student.named_modules()]
    with torch.inference_mode():
        expected = student(latent)[0]
    handle = installer()(student, conv_channel_pairs={(8, 8)})
    assert handle.enabled and handle.convs
    with torch.inference_mode():
        actual = student(latent)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert topology == [(name, type(layer)) for name, layer in student.named_modules()]
    assert pointers == {name: value.data_ptr() for name, value in student.state_dict().items()}
    for name, value in student.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    handle.disable()
    assert not handle.enabled
    with torch.inference_mode():
        torch.testing.assert_close(student(latent)[0], expected, rtol=0, atol=0)
    handle.enable()
    with torch.inference_mode():
        torch.testing.assert_close(student(latent)[0], expected, rtol=0, atol=0)
    handle.remove()


def test_active_adapter_rejects_gradients_but_disable_restores_training(student):
    handle = installer()(student)
    latent = torch.randn(1, 48, 2, 1, 1, requires_grad=True)
    with pytest.raises(RuntimeError, match="(?i)(grad|inference|training)"):
        student(latent)
    student.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="(?i)(eval|training|inference)"):
        student(latent)
    handle.disable()
    output = student(latent)[0]
    output[:, :, -1:].square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert student.last_weight.grad is not None
    assert torch.isfinite(student.last_weight.grad).all()


def test_duplicate_install_requires_remove(student):
    install = installer()
    handle = install(student)
    with pytest.raises(ValueError, match="(?i)already"):
        install(student)
    handle.disable()
    with pytest.raises(ValueError, match="(?i)already"):
        install(student)
    handle.remove()
    replacement = install(student, conv_channel_pairs=set(), fuse_norm=False,
                          fuse_upsample=False, structural=False)
    assert not replacement.convs
    # Repeated cleanup of an old handle must not unregister its replacement.
    handle.remove()
    with pytest.raises(ValueError, match="(?i)already"):
        install(student)
    assert replacement.enabled
    replacement.remove()


def test_export_from_accelerated_model_remains_portable(student, source_root, tmp_path):
    install = installer()
    handle = install(student)
    path = tmp_path / "student.pt"
    export_student(student, path, source_root)
    restored, artifact = load_student(path, source_root)
    assert artifact["width"] == {"stages": [8, 8, 8, 4, 2], "hidden": {}}
    assert list(student.state_dict()) == list(restored.state_dict())
    latent = torch.randn(1, 48, 2, 1, 2)
    with torch.inference_mode():
        torch.testing.assert_close(student(latent)[0], restored(latent)[0], rtol=0, atol=0)
    handle.remove()
