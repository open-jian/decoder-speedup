import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from decoder_compress.config import Config, WidthConfig, load_config, from_dict
from decoder_compress.data import BatchStream, VideoDataset, assert_disjoint, split_manifest, vidgen_source_id
from decoder_compress.evaluation import quality_metrics


@pytest.mark.parametrize("width", [WidthConfig([8, 3, 4, 2, 1]), WidthConfig([8, 8]), WidthConfig([0, 8, 8, 4, 2]),
                                 WidthConfig(hidden={"no.such.block": 32}), WidthConfig([True, 8, 8, 4, 2])])
def test_invalid_width(width):
    with pytest.raises(ValueError):
        width.validate()


def test_config_strict_and_paths(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"training": {"max_step": 100}})
    file = tmp_path / "config.yaml"
    file.write_text("width:\n  stages: [8, 8, 8, 4, 2]\nmodel:\n  source: ../Wan2.2\n")
    assert load_config(file).model.source == str(tmp_path.parent / "Wan2.2")
    file.write_text("version: 1\nversion: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(file)


def test_split_by_original_source():
    rows = [{"path": f"{source}-{clip}.mp4", "source_id": str(source)} for source in range(100) for clip in range(3)]
    train, val = split_manifest(rows, 0.2)
    assert len(train) + len(val) == 300
    assert not {r["source_id"] for r in train} & {r["source_id"] for r in val}
    assert (train, val) == split_manifest(list(reversed(rows)), 0.2)
    assert vidgen_source_id("abcdefghijk-Scene-001.mp4") == "abcdefghijk"
    with pytest.raises(ValueError):
        vidgen_source_id("unknown.mp4")


class Dataset:
    identity = {"manifest_sha256": "fixture"}
    def __len__(self):
        return 5
    def get(self, index, seed):
        return torch.tensor([index, seed % 100000])


def test_batch_stream_resume_epoch_boundary():
    stream = BatchStream(Dataset(), 3, 42)
    stream.next()
    state = stream.state_dict()
    expected = [stream.next() for _ in range(4)]
    resumed = BatchStream(Dataset(), 3, 42)
    resumed.load_state_dict(state)
    for batch in expected:
        torch.testing.assert_close(batch, resumed.next())


def test_video_decode_and_quality(tmp_path):
    path = tmp_path / "clip.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert out.isOpened()
    for i in range(12):
        out.write(np.full((32, 32, 3), i * 15, dtype=np.uint8))
    out.release()
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"path": "clip.avi", "source_id": "one"}) + "\n")
    cfg = Config().data
    cfg.root, cfg.frames, cfg.height, cfg.width = str(tmp_path), 5, 16, 16
    data = VideoDataset(cfg, manifest)
    clip = data.get(0, 24)
    assert clip.shape == (3, 5, 16, 16)
    torch.testing.assert_close(clip, data.get(0, 24))
    assert clip.min() >= -1 and clip.max() <= 1
    result = quality_metrics(clip[None], clip[None])
    assert result["psnr_db"] == 120 and abs(result["ssim"] - 1) < 1e-6
    with pytest.raises(ValueError, match="overlapping"):
        assert_disjoint(data, data)
    cfg.frames = 17
    with pytest.raises(RuntimeError, match="short"):
        data.get(0, 0)


def test_vidgen_manifest_excludes_unfinished_extraction(tmp_path):
    from decoder_compress.data import make_manifests
    for folder in ("videos/package", "extraction_in_progress/package"):
        (tmp_path / folder).mkdir(parents=True)
    for i in range(100):
        (tmp_path / "videos/package" / f"{i:011d}-Scene-0001.mp4").touch()
    (tmp_path / "extraction_in_progress/package/unfinished.mp4").touch()
    train, val = make_manifests(tmp_path, tmp_path / "manifests", validation_fraction=0.2)
    assert len(train) + len(val) == 100
    assert all(row["path"].startswith("videos/") for row in train + val)
