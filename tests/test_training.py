import copy
import types
import pytest
import torch
import torch.nn.functional as F
from decoder_speedup.config import Config, WidthConfig
from decoder_speedup.data import BatchStream
from decoder_speedup.models.wan22.adapter import build_student
from decoder_speedup.training.trainer import Trainer, seed_all
from test_adapter import tiny_teacher


class SyntheticVideos:
    identity = {"manifest_sha256": "synthetic-regression-fixture", "rows": 5}
    def __len__(self):
        return 5
    def get(self, index, seed):
        generator = torch.Generator().manual_seed(seed % (2**63 - 1))
        return torch.randn(3, 5, 16, 16, generator=generator).tanh()


def fixture(source):
    seed_all(17)
    teacher = tiny_teacher(source)
    @torch.no_grad()
    def prepare(video, feature_paths):
        latent = F.adaptive_avg_pool3d(video, (2, 1, 1)).repeat(1, 16, 1, 1, 1)
        return latent, teacher.decoder(latent, feature_paths)[1]
    teacher.prepare = prepare
    cfg = Config()
    cfg.width = WidthConfig([8, 4, 4, 2, 2])
    cfg.runtime.device, cfg.runtime.precision = "cpu", "fp32"
    cfg.data.frames, cfg.data.height, cfg.data.width = 5, 16, 16
    cfg.loss.lpips = 0
    cfg.training.reconstruction_updates = 1
    cfg.training.adversarial_updates = 2
    cfg.training.accumulation = 2
    cfg.training.discriminator_updates = 2
    cfg.training.discriminator_channels = 2
    cfg.training.ema_decay = 0.9
    student, _ = build_student(source, cfg.width, teacher, "teacher_prefix")
    stream = BatchStream(SyntheticVideos(), 1, 23)
    return Trainer(student, teacher, stream, cfg)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_gan_accumulation_and_complete_resume(source, tmp_path):
    trainer = fixture(source)
    initial = trainer.student.last_weight.detach().clone()
    first = trainer.step()
    assert first["generator_updates"] == 1 and first["discriminator_updates"] == 0
    assert first["microbatches"] == 2 and first["stage"] == "reconstruction"
    assert not torch.equal(initial, trainer.student.last_weight)
    second = trainer.step()
    assert second["discriminator_updates"] == 2 and second["microbatches"] == 4
    assert second["gan_weight"] > 0
    trainer.save(tmp_path / "resume.pt")
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert state["format"] == "decoder-speedup-training-v1"
    state["format"] = "decoder-compress-training-v1"
    torch.save(state, tmp_path / "resume.pt")
    reference = trainer.step()
    resumed = fixture(source)
    resumed.resume(tmp_path / "resume.pt")
    result = resumed.step()
    assert result == reference
    assert result["generator_updates"] == 3 and result["discriminator_updates"] == 4
    for attr in ("student", "alignment", "discriminator", "optimizer", "d_optimizer"):
        assert_tree_equal(getattr(trainer, attr).state_dict(), getattr(resumed, attr).state_dict())
    assert_tree_equal(trainer.ema, resumed.ema)
    assert trainer.stream.state_dict() == resumed.stream.state_dict()
    assert all(p.grad is None for p in trainer.teacher.decoder.parameters())
    with pytest.raises(StopIteration):
        resumed.step()
    raw = resumed.student.last_weight.detach().clone()
    with resumed.ema_weights():
        assert not torch.equal(raw, resumed.student.last_weight)
    torch.testing.assert_close(raw, resumed.student.last_weight, rtol=0, atol=0)


def test_resume_rejects_changed_recipe(source, tmp_path):
    trainer = fixture(source)
    trainer.step()
    trainer.save(tmp_path / "state.pt")
    changed = fixture(source)
    changed.config.training.learning_rate *= 2
    with pytest.raises(ValueError, match="matching config"):
        changed.resume(tmp_path / "state.pt")
