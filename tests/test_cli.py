import json
import cv2
import numpy as np
import torch
import yaml
from decoder_compress import cli
from decoder_compress.provenance import source_identity
from decoder_compress.export import load_student
from test_training import fixture


def test_cli_training_export_evaluate(source, source_root, tmp_path, monkeypatch):
    setup = fixture(source)
    cfg = setup.config
    cfg.training.accumulation = 1
    cfg.training.adversarial_updates = 1
    cfg.training.eval_every = 1
    cfg.training.eval_clips = 1
    cfg.model.source = str(source_root)
    cfg.model.weights = str(tmp_path / "teacher.fixture")
    (tmp_path / "teacher.fixture").write_text("tiny synthetic teacher")
    cfg.data.root = str(tmp_path)
    for split in ("train", "val"):
        video = tmp_path / f"{split}.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
        assert writer.isOpened()
        for i in range(9):
            writer.write(np.full((32, 32, 3), i * 24, dtype=np.uint8))
        writer.release()
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps({"path": video.name, "source_id": split}) + "\n")
        setattr(cfg.data, f"{split}_manifest", str(manifest))
    cfg.output = str(tmp_path / "run")
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()))
    monkeypatch.setattr(cli, "setup", lambda config: (setup.teacher, setup.student, {"source": source_identity(source_root)}))
    cli.main(["train", str(config_path)])
    assert (tmp_path / "run/last.pt").exists()
    assert (tmp_path / "run/evaluation-final.json").exists()
    rows = [json.loads(s) for s in (tmp_path / "run/train.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]["stage"] == "adversarial"
    exported = tmp_path / "export.pt"
    cli.main(["export", str(tmp_path / "run/last.pt"), "--source", str(source_root), "--output", str(exported)])
    model, metadata = load_student(exported, source_root)
    assert metadata["provenance"]["generator_updates"] == 2
    with torch.no_grad():
        assert model(torch.zeros(1, 48, 2, 1, 1))[0].shape == (1, 3, 5, 16, 16)
    monkeypatch.setattr(cli, "WanTeacher", lambda *args, **kwargs: setup.teacher)
    quality_path = tmp_path / "quality.json"
    cli.main(["evaluate", str(config_path), "--student", str(exported), "--output", str(quality_path)])
    quality = json.loads(quality_path.read_text())
    assert quality["clips"] == 1 and set(quality["means"]) == {"teacher", "student"}
    timing_path = tmp_path / "timing.json"
    cli.main(["benchmark", str(config_path), "--student", str(exported), "--output", str(timing_path),
              "--warmup", "1", "--repeats", "3"])
    timing = json.loads(timing_path.read_text())
    assert len(timing["native_teacher"]["samples_ms"]) == 3
    assert timing["runtime_student"]["speedup_vs_native"] > 0
