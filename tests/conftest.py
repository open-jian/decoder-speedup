import os
from pathlib import Path
import pytest
import torch
from decoder_compress.models.wan22.adapter import load_source


@pytest.fixture(scope="session", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="session")
def source_root():
    default = Path(__file__).resolve().parents[2] / "Wan2.2"
    path = Path(os.environ.get("WAN22_SOURCE", default))
    if not (path / "wan/modules/vae2_2.py").exists():
        pytest.skip("Set WAN22_SOURCE to run official Wan adapter integration tests")
    return path


@pytest.fixture(scope="session")
def source(source_root):
    return load_source(source_root)
