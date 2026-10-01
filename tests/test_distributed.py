from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from decoder_speedup.training.budget import resolve_budget
from decoder_speedup.config import Config
from decoder_speedup.models.wan22.adapter import load_source
from test_training import fixture, assert_tree_equal


def worker(rank, root, source_root):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method=f'file://{root / "rendezvous"}', rank=rank, world_size=2)
    try:
        source = load_source(source_root)
        trainer = fixture(source)
        first, second = trainer.step(), trainer.step()
        trainer.save(root / 'parallel.pt')
        reference = trainer.step()
        resumed = fixture(source)
        resumed.resume(root / 'parallel.pt')
        assert resumed.step() == reference
        for name in ('student', 'alignment', 'optimizer', 'discriminator', 'd_optimizer'):
            assert_tree_equal(getattr(trainer, name).state_dict(), getattr(resumed, name).state_dict())
        assert_tree_equal(trainer.ema, resumed.ema)
        # A single-rank checkpoint preserves the same global data position and batch.
        migrated = fixture(source)
        migrated.resume(root / 'single.pt')
        migrated_second = migrated.step()
        torch.save({'first':first, 'second':second, 'migrated_second':migrated_second,
                    'student':migrated.student.state_dict(), 'ema':migrated.ema,
                    'stream':migrated.stream.state_dict(), 'plan':migrated.training_plan}, root / f'rank-{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_rec_gan_resume_and_single_rank_migration(source, source_root, tmp_path):
    baseline = fixture(source)
    first = baseline.step()
    baseline.save(tmp_path / 'single.pt')
    second = baseline.step()
    mp.spawn(worker, args=(str(tmp_path), str(source_root)), nprocs=2, join=True)
    one = torch.load(tmp_path / 'rank-0.pt', weights_only=False)
    two = torch.load(tmp_path / 'rank-1.pt', weights_only=False)
    assert_tree_equal(one, two)
    for result, expected in ((one['first'], first), (one['second'], second), (one['migrated_second'], second)):
        for name, value in expected.items():
            if isinstance(value, float):
                assert result[name] == pytest.approx(value, rel=1e-5, abs=1e-6)
            else:
                assert result[name] == value
    for name, value in baseline.student.state_dict().items():
        torch.testing.assert_close(one['student'][name], value, rtol=1e-5, atol=1e-6)
    assert one['stream'] == baseline.stream.state_dict()
    assert one['plan']['world_size'] == 2 and one['plan']['effective_batch'] == 2
    assert one['plan']['changes'][-1]['event'] == 'world_size_change'


def test_four_rank_budget_keeps_global_batch():
    cfg = Config()
    cfg.training.accumulation = 8
    cfg.training.reconstruction_updates = None
    cfg.training.reconstruction_epochs = 100
    plan = resolve_budget(cfg, 10000, world_size=4)
    assert plan['local_accumulation'] == 2 and plan['effective_batch'] == 8
    assert cfg.training.reconstruction_updates == 125000
    with pytest.raises(ValueError, match='divisible'):
        resolve_budget(cfg, 10000, world_size=3)


def test_framework_migration_remains_strict_for_data_and_recipe(source, tmp_path):
    original = fixture(source)
    original.provenance = {'framework_files': {'trainer.py':'old'}, 'data':{'identity':'same'}}
    original.step()
    original.save(tmp_path / 'state.pt')
    changed = fixture(source)
    changed.provenance = {'framework_files': {'trainer.py':'new'}, 'data':{'identity':'same'}}
    with pytest.raises(ValueError, match='matching config'):
        changed.resume(tmp_path / 'state.pt')
    changed.resume(tmp_path / 'state.pt', allow_framework_change=True)
    assert changed.training_plan['changes'][-1]['event'] == 'framework_change'
    assert_tree_equal(original.optimizer.state_dict(), changed.optimizer.state_dict())
    changed.provenance['data']['identity'] = 'different'
    with pytest.raises(ValueError, match='matching config'):
        changed.resume(tmp_path / 'state.pt', allow_framework_change=True)


def test_stop_request_saves_complete_update(source, tmp_path):
    trainer = fixture(source, reconstruction_updates=5, adversarial_updates=0)
    (tmp_path / 'stop.request').touch()
    trainer.fit(tmp_path)
    state = torch.load(tmp_path / 'last.pt', weights_only=False)
    assert state['updates'] == 1 and state['stream']['cursor'] == 2
    assert state['config']['training']['reconstruction_updates'] == 5
