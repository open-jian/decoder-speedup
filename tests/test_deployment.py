"""Deployment input/lifetime guards; real CUDA decoding is checked separately."""
import threading

import pytest
import torch

from decoder_speedup import deployment
from decoder_speedup.deployment import Wan22CompressedDecoder


@pytest.fixture
def decoder():
    # Validation and cleanup require no model allocation or CUDA initialization.
    result = Wan22CompressedDecoder.__new__(Wan22CompressedDecoder)
    result.device = torch.device("cuda:0")
    result._lock = threading.RLock()
    result._closed = False
    result._graph = result._graph_key = result._handle = result.model = None
    return result


def test_constructor_rejects_cpu_before_loading():
    with pytest.raises(ValueError, match="CUDA device"):
        Wan22CompressedDecoder("missing.pt", "missing-source", device="cpu")


def test_loading_inside_inference_mode_keeps_normal_weight_versions(monkeypatch):
    def load(*args):
        assert not torch.is_inference_mode_enabled()
        return torch.nn.Linear(2, 2), {"example": True}
    monkeypatch.setattr(deployment, "load_student", load)
    with torch.inference_mode():
        result = Wan22CompressedDecoder("checkpoint", "source", use_winograd=False)
    assert not result.model.weight.is_inference()
    assert isinstance(result.model.weight._version, int)
    assert not result.model.training
    assert not result.model.weight.requires_grad
    result.close()


@pytest.mark.parametrize("value,error,match", [
    (torch.empty(48, 1, 1, 1), TypeError, "list"),
    ([None], TypeError, "torch.Tensor"),
    ([torch.empty(16, 1, 1, 1)], ValueError, "48,T,H,W"),
    ([torch.empty(48, 0, 1, 1)], ValueError, "nonempty"),
    ([torch.empty(1, 48, 1, 1, 1)], ValueError, "48,T,H,W"),
    ([torch.ones(48, 1, 1, 1, dtype=torch.int64)], ValueError, "dtype"),
    ([torch.empty(48, 1, 1, 1)], ValueError, "must match decoder device"),
])
def test_decode_rejects_invalid_inputs_before_cuda(decoder, value, error, match):
    with pytest.raises(error, match=match):
        decoder.decode(value)


def test_empty_list_matches_official_contract(decoder):
    assert decoder.decode([]) == []


def test_close_releases_graph_before_adapter_and_rejects_reuse(decoder):
    events = []
    class Graph:
        def close(self):
            events.append("graph")
    class Handle:
        def remove(self):
            events.append("handle")
    decoder._graph = Graph()
    decoder._handle = Handle()
    decoder.close()
    decoder.close()
    assert events == ["graph", "handle"]
    assert decoder.model is decoder._graph is decoder._handle is None
    with pytest.raises(RuntimeError, match="closed"):
        decoder.decode([])
