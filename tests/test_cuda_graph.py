"""CPU checks for graph safety guards; GPU replay is validated separately."""

import pytest
import torch

from decoder_speedup.cuda_graph import CapturedDecoder, _model_signature, capture_decoder


def test_cpu_capture_is_rejected_without_initializing_cuda(monkeypatch):
    def unexpected_cuda_initialization():
        raise AssertionError("CPU rejection attempted to initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected_cuda_initialization)
    model = torch.nn.Linear(3, 3).eval()
    with pytest.raises(ValueError, match="CUDA latent"):
        capture_decoder(model, torch.zeros(1, 48, 1, 1, 1))


@pytest.mark.parametrize("warmup", [0, -1, True])
def test_capture_requires_positive_integer_warmup(warmup):
    with pytest.raises(ValueError, match="warmup"):
        capture_decoder(torch.nn.Identity().eval(), torch.zeros(1), warmup=warmup)


@pytest.mark.parametrize("change", ["weight", "buffer", "storage", "hook", "forward", "training", "precision"])
def test_stale_model_guard_requires_recapture(change, monkeypatch):
    model = torch.nn.Linear(3, 3).eval()
    model.register_buffer("scale", torch.ones(3))
    # Exercise only the CPU model-validity guard. No CUDA graph or GPU execution
    # is mocked; replay correctness belongs to the separate real-GPU checks.
    captured = CapturedDecoder.__new__(CapturedDecoder)
    captured.model = model
    captured.kernel_handle = None
    captured.closed = False
    captured._signature = _model_signature(model)
    captured._check_model()
    with torch.no_grad():
        if change == "weight":
            model.weight.add_(0.01)
        elif change == "buffer":
            model.scale.mul_(2)
        elif change == "storage":
            model.weight = torch.nn.Parameter(model.weight.clone())
        elif change == "hook":
            model.register_forward_hook(lambda module, inputs, output: output)
        elif change == "forward":
            model.forward = lambda x: x
        elif change == "training":
            model.train()
        elif change == "precision":
            setting = "allow_bf16_reduced_precision_reduction"
            monkeypatch.setattr(torch.backends.cuda.matmul, setting,
                                not getattr(torch.backends.cuda.matmul, setting))
    with pytest.raises(RuntimeError, match="recapture"):
        captured._check_model()


def test_inference_created_weights_lack_required_mutation_counters():
    with torch.inference_mode():
        model = torch.nn.Linear(3, 3).eval()
    with pytest.raises(ValueError, match="outside inference_mode"):
        _model_signature(model)
