import json
from pathlib import Path
import pytest
import torch
import yaml
from decoder_speedup.config import Config, from_dict
from decoder_speedup.training.budget import resolve_budget
from decoder_speedup import cli
from test_training import fixture, assert_tree_equal
from test_tracking import sdk, WandbTracker, tracked_config


def test_epoch_budget_uses_samples_and_complete_updates():
    cfg = from_dict({'training': {'reconstruction_updates': None, 'reconstruction_epochs': 100,
                                  'accumulation': 8, 'lr_schedule': 'constant'}})
    plan = resolve_budget(cfg, 10000)
    assert plan['effective_batch'] == 8
    assert plan['requested_reconstruction_samples'] == 1000000
    assert plan['rounded_extra_samples'] == 0
    assert cfg.training.reconstruction_updates == 125000
    assert cfg.training.reconstruction_epochs is None
    rounded = from_dict({'training': {'reconstruction_updates': None, 'reconstruction_epochs': 3,
                                      'accumulation': 8}})
    plan = resolve_budget(rounded, 11)
    assert rounded.training.reconstruction_updates == 5 and plan['rounded_extra_samples'] == 7
    with pytest.raises(ValueError, match='null'):
        from_dict({'training': {'reconstruction_epochs': 100}})
    with pytest.raises(ValueError, match='constant'):
        from_dict({'training': {'lr_schedule': 'cosine'}})


def test_turbo_public_preset():
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / 'configs/wan22/width.yaml').read_text())
    cfg = from_dict(raw)
    assert cfg.width.stages == [512, 512, 256, 64, 32] and not cfg.width.hidden
    assert cfg.data.batch_size == 1 and cfg.training.accumulation == 8
    assert cfg.training.reconstruction_updates is None and cfg.training.reconstruction_epochs == 100
    assert cfg.training.adversarial_updates == 0 and cfg.training.lr_schedule == 'constant'
    assert cfg.training.learning_rate == cfg.training.discriminator_lr == 1e-4
    assert cfg.training.epsilon == 1e-15
    assert cfg.training.betas == [0.9, 0.95] and cfg.training.ema_decay == .999


def test_gan_transition_preserves_generator_and_resumes_exactly(source, tmp_path):
    rec = fixture(source, reconstruction_updates=10, adversarial_updates=0)
    rec.tracking_state = {'run_id': 'same-run'}
    for _ in range(3):
        rec.step()
    rec.save(tmp_path / 'rec.pt')
    gan = fixture(source, reconstruction_updates=10, adversarial_updates=0)
    gan.resume(tmp_path / 'rec.pt', start_gan_updates=2)
    for attr in ('student', 'alignment', 'optimizer'):
        assert_tree_equal(getattr(rec, attr).state_dict(), getattr(gan, attr).state_dict())
    assert_tree_equal(rec.ema, gan.ema)
    assert rec.stream.state_dict() == gan.stream.state_dict()
    assert gan.tracking_state == rec.tracking_state
    assert gan.config.training.reconstruction_updates == 3 and gan.config.training.adversarial_updates == 2
    assert gan.updates == 3 and gan.d_updates == 0
    assert gan.optimizer.param_groups[0]['lr'] == 1e-4
    assert gan.training_plan['changes'][-1]['event'] == 'start_gan'
    first = gan.step()
    assert first['stage'] == 'adversarial' and first['generator_updates'] == 4
    assert first['discriminator_updates'] == 2
    gan.save(tmp_path / 'gan.pt')
    reference = gan.step()
    resumed = fixture(source, reconstruction_updates=3, adversarial_updates=2)
    resumed.resume(tmp_path / 'gan.pt')
    assert resumed.step() == reference
    for attr in ('student', 'alignment', 'optimizer', 'discriminator', 'd_optimizer'):
        assert_tree_equal(getattr(gan, attr).state_dict(), getattr(resumed, attr).state_dict())
    assert_tree_equal(gan.ema, resumed.ema)
    with pytest.raises(ValueError, match='before any GAN'):
        resumed.resume(tmp_path / 'gan.pt', start_gan_updates=2)


def test_budget_extension_keeps_state_and_rejects_other_changes(source, tmp_path):
    rec = fixture(source, reconstruction_updates=3, adversarial_updates=0)
    rec.step()
    rec.save(tmp_path / 'state.pt')
    reference = rec.step()
    resumed = fixture(source, reconstruction_updates=3, adversarial_updates=0)
    resumed.resume(tmp_path / 'state.pt', extend_reconstruction_updates=5)
    assert resumed.config.training.reconstruction_updates == 8
    assert resumed.step() == reference
    assert resumed.discriminator is None
    changed = fixture(source, reconstruction_updates=3, adversarial_updates=0)
    changed.config.training.learning_rate *= 2
    with pytest.raises(ValueError, match='matching config'):
        changed.resume(tmp_path / 'state.pt', start_gan_updates=2)
    changed.config.training.learning_rate /= 2
    changed.config.loss.feature *= 2
    with pytest.raises(ValueError, match='matching config'):
        changed.resume(tmp_path / 'state.pt', extend_reconstruction_updates=5)


def test_review_stop_keeps_plan_pending_window_and_lr(source, tmp_path, sdk):
    trainer = fixture(source, reconstruction_updates=8, adversarial_updates=0)
    trainer.config.wandb = tracked_config().wandb
    trainer.config.wandb.log_every = 50
    with WandbTracker(trainer.config, tmp_path) as tracker:
        trainer.tracking_state = tracker.state
        trainer.fit(tmp_path, tracker=tracker, stop_after_updates=3)
        tracker.mark_complete(trainer)
        assert not tracker.run.summary['training_complete']
        assert tracker.run.summary['session_stop_reason'] == 'review'
    state = torch.load(tmp_path / 'last.pt', weights_only=False)
    assert state['updates'] == 3 and state['config']['training']['reconstruction_updates'] == 8
    assert state['tracking']['window']['count'] == 3
    resumed = fixture(source, reconstruction_updates=8, adversarial_updates=0)
    resumed.resume(tmp_path / 'last.pt')
    assert resumed.optimizer.param_groups[0]['lr'] == 1e-4
    resumed.fit(tmp_path, stop_after_updates=4)
    assert resumed.updates == 4
    with pytest.raises(ValueError, match='exceed'):
        resumed.fit(tmp_path, stop_after_updates=4)


def test_legacy_config_defaults_are_normalized_on_resume(source, tmp_path):
    trainer = fixture(source)
    trainer.step()
    trainer.save(tmp_path / 'legacy.pt')
    state = torch.load(tmp_path / 'legacy.pt', weights_only=False)
    state['config']['training'].pop('lr_schedule')
    state['config']['training'].pop('reconstruction_epochs')
    state.pop('training_plan')
    torch.save(state, tmp_path / 'legacy.pt')
    resumed = fixture(source)
    resumed.resume(tmp_path / 'legacy.pt')
    assert resumed.updates == trainer.updates
    assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())


def test_plan_command_does_not_load_models_or_start_tracking(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(''.join(json.dumps({'path': f'{i}.mp4', 'source_id': str(i)}) + '\n' for i in range(5)))
    cfg = Config()
    cfg.data.root, cfg.data.train_manifest = str(tmp_path), str(manifest)
    cfg.training.reconstruction_updates = None
    cfg.training.reconstruction_epochs = 100
    cfg.training.accumulation = 8
    config_path = tmp_path / 'epoch.yaml'
    config_path.write_text(yaml.safe_dump(cfg.to_dict()))
    def forbidden(*args, **kwargs):
        raise AssertionError('Model/SDK must not initialize in plan')
    monkeypatch.setattr(cli, 'setup', forbidden)
    monkeypatch.setattr(cli, 'WandbTracker', forbidden)
    cli.main(['plan', str(config_path)])
    report = json.loads(capsys.readouterr().out)
    assert report['reconstruction_updates'] == 63 and report['rounded_extra_samples'] == 4
