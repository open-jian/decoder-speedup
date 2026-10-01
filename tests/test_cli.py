import json
import cv2
import numpy as np
import torch
import yaml
from decoder_speedup import cli
from decoder_speedup.provenance import source_identity
from decoder_speedup.export import load_student
from test_training import fixture
from test_tracking import sdk, tracked_config


def test_cli_training_export_evaluate(source, source_root, tmp_path, monkeypatch, sdk):
    setup = fixture(source)
    cfg = setup.config
    cfg.wandb = tracked_config().wandb
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
    original_benchmark = cli.benchmark_decoder
    def checked_benchmark(*args, **kwargs):
        assert len(sdk[0]) == 2  # No benchmark W&B process during any of the three timers.
        return original_benchmark(*args, **kwargs)
    monkeypatch.setattr(cli, "benchmark_decoder", checked_benchmark)
    cli.main(["benchmark", str(config_path), "--student", str(exported), "--output", str(timing_path),
              "--warmup", "1", "--repeats", "3"])
    timing = json.loads(timing_path.read_text())
    assert len(timing["native_teacher"]["samples_ms"]) == 3
    assert timing["runtime_student"]["speedup_vs_native"] > 0
    assert [call["job_type"] for call in sdk[0]] == ["decoder-recovery", "quality-eval", "decode-benchmark"]
    assert sdk[1][-1].history[0]["decode/runtime_student_speedup_vs_native"] > 0
    assert sdk[1][-1].summary["origin_training_run_id"] == sdk[1][0].id
    quality_steps = [row["progress/generator_updates"] for row in sdk[1][0].history if "quality/clips" in row]
    assert quality_steps == [1, 2]  # Final evaluation is reused, not computed/uploaded twice.
    # A reviewed reconstruction checkpoint enters GAN with full state, then writes a resumable plan.
    cli.main(['train', str(config_path), '--resume', str(tmp_path / 'run/checkpoint-00000001.pt'),
              '--start-gan-updates', '2', '--stop-after-updates', '2'])
    staged = torch.load(tmp_path / 'run/last.pt', weights_only=False)
    assert staged['updates'] == 2 and staged['config']['training']['adversarial_updates'] == 2
    assert staged['tracking']['run_id'] == sdk[1][0].id
    assert not sdk[1][-1].summary['training_complete']
    assert staged['training_plan']['changes'][-1]['event'] == 'start_gan'
    cli.main(['train', str(tmp_path / 'run/config.resolved.yaml'), '--resume', str(tmp_path / 'run/last.pt')])
    finished = torch.load(tmp_path / 'run/last.pt', weights_only=False)
    assert finished['updates'] == 3 and sdk[1][-1].summary['training_complete']
