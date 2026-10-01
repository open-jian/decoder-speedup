import copy
import json
from pathlib import Path
import sys
import types
import pytest
import torch
from decoder_speedup.config import Config, WandbConfig, from_dict
from decoder_speedup.tracking import WandbTracker, check_connection
from test_training import fixture


class FakeRun:
    def __init__(self, arguments):
        self.id = arguments["id"]
        self.url = f"https://wandb.ai/{arguments['entity']}/{arguments['project']}/runs/{self.id}"
        self.summary = {}
        self.history = []
        self.definitions = []
        self.exit_codes = []

    def log(self, metrics):
        self.history.append(dict(metrics))

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)


@pytest.fixture
def sdk(monkeypatch):
    calls, runs = [], []
    def init(**kwargs):
        calls.append(kwargs)
        run = FakeRun(kwargs)
        runs.append(run)
        return run
    module = types.SimpleNamespace(init=init, Settings=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return calls, runs


def tracked_config():
    return Config(wandb=WandbConfig(enabled=True, entity="miaoyin-uta", project="vae-speedup", log_every=1))


def test_disabled_never_initializes_sdk(sdk, tmp_path):
    with WandbTracker(Config(), tmp_path) as tracker:
        assert tracker.state == {} and tracker.run is None
    assert not sdk[0] and not list(tmp_path.iterdir())


def test_only_allowed_configuration_is_uploaded(sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "SECRET_NOT_FOR_LOGGING")
    cfg = tracked_config()
    cfg.model.weights = "/private/teacher.pth"
    cfg.data.root = "/private/dataset"
    provenance = {"initialization": {"path": "/private/student.pt"}, "teacher_sha256": "hash"}
    with WandbTracker(cfg, tmp_path, provenance) as tracker:
        tracker.log_validation({"clips": 1, "means": {"teacher": {"psnr_db": 35.0}, "student": {"psnr_db": 33.0}},
                                "per_clip": [{"path": "/private/video.mp4"}]}, 10)
    arguments = sdk[0][0]
    uploaded = json.dumps(arguments["config"])
    assert "SECRET_NOT_FOR_LOGGING" not in uploaded and "/private/" not in uploaded
    assert arguments["save_code"] is False
    assert arguments["settings"]["disable_code"] and arguments["settings"]["disable_git"]
    assert sdk[1][0].history[0]["validation/student_minus_teacher/psnr_db"] == -2.0
    assert sdk[1][0].exit_codes == [0]
    assert json.loads((tmp_path / "wandb-run.json").read_text())["run_id"] == tracker.state["run_id"]


def test_training_counts_checkpoint_resume_and_log_failure(source, sdk, tmp_path):
    trainer = fixture(source)
    trainer.config.wandb = tracked_config().wandb
    with WandbTracker(trainer.config, tmp_path) as tracker:
        trainer.tracking_state = tracker.state
        trainer.fit(tmp_path, tracker=tracker)
        tracker.mark_complete(trainer)
    history = sdk[1][0].history
    assert [row["progress/generator_updates"] for row in history] == [1, 2, 3]
    assert [row["progress/discriminator_updates"] for row in history] == [0, 2, 4]
    assert history[-1]["progress/microbatches"] == 6
    assert history[-1]["progress/fraction"] == 1
    assert history[0]["optim/generator_lr"] == trainer.config.training.learning_rate
    assert history[1]["train/gan_active"] == 1 and "train/d_loss" in history[1]
    saved = torch.load(tmp_path / "last.pt", weights_only=False)
    assert saved["tracking"]["run_id"] == tracker.state["run_id"]
    resumed = fixture(source)
    resumed.config.wandb = copy.deepcopy(trainer.config.wandb)
    resumed.config.wandb.log_every = 20
    resumed.resume(tmp_path / "last.pt")
    with WandbTracker(resumed.config, tmp_path, state=resumed.tracking_state):
        pass
    assert sdk[0][-1]["id"] == tracker.state["run_id"] and sdk[0][-1]["resume"] == "must"
    with pytest.raises(RuntimeError, match="simulated"):
        with WandbTracker(tracked_config(), tmp_path / "failure"):
            raise RuntimeError("simulated training failure")
    assert sdk[1][-1].exit_codes == [1]


def test_connection_check_has_no_training_metrics(sdk, tmp_path):
    result = check_connection("miaoyin-uta", "vae-speedup", tmp_path)
    assert result["connection_ok"] and not result["formal_training_started"]
    assert sdk[0][0]["job_type"] == "connection-check"
    assert sdk[1][0].history == [{"setup/connection_ok": 1, "setup/optimizer_updates": 0}]
    assert json.loads((tmp_path / "connection-check.json").read_text())["url"] == result["url"]


def test_resume_does_not_silently_change_project_or_offline_identity(sdk, tmp_path):
    cfg = tracked_config()
    with WandbTracker(cfg, tmp_path) as tracker:
        state = tracker.state
    cfg.wandb.project = "another-project"
    with pytest.raises(ValueError, match="another entity/project"):
        WandbTracker(cfg, tmp_path, state=state)
    cfg = tracked_config()
    cfg.wandb.mode = "offline"
    state = dict(state, mode="offline")
    with WandbTracker(cfg, tmp_path, state=state) as offline:
        assert offline.state["run_id"] != state["run_id"]
        assert offline.run.summary["resumed_from_offline_run"] == state["run_id"]
    assert sdk[0][-1]["resume"] is None


def test_credentials_cannot_be_put_in_wandb_config():
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"wandb": {"api_key": "must-not-be-config"}})
    with pytest.raises(ValueError, match="separately"):
        from_dict({"wandb": {"enabled": True, "entity": "miaoyin-uta/vae-speedup"}})
