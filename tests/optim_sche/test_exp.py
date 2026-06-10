import pytest

torch = pytest.importorskip("torch")

from speechain.optim_sche.exp import ExponentDecayLr


@pytest.fixture
def exp_scheduler():
    model = torch.nn.Linear(4, 4)
    return ExponentDecayLr(
        optim_type="Adam",
        optim_conf={"lr": 1e-3},
        model=model,
        decay_factor=0.9,
        use_amp=False,
    )


class TestExponentDecayLr:
    def test_instantiation(self, exp_scheduler):
        assert exp_scheduler is not None
        assert exp_scheduler.decay_factor == 0.9

    def test_no_decay_at_epoch_one(self, exp_scheduler):
        base_lr = exp_scheduler.get_lr()
        lr_epoch1 = exp_scheduler.update_lr(0, 1)
        assert abs(lr_epoch1 - base_lr) < 1e-10

    def test_decay_over_epochs(self, exp_scheduler):
        lr_epoch1 = exp_scheduler.update_lr(0, 1)
        lr_epoch2 = exp_scheduler.update_lr(0, 2)
        assert lr_epoch2 < lr_epoch1

    def test_decay_monotonic(self, exp_scheduler):
        lrs = [exp_scheduler.update_lr(0, e) for e in range(1, 6)]
        assert all(lrs[i] > lrs[i + 1] for i in range(len(lrs) - 1))

    def test_extra_repr(self, exp_scheduler):
        repr_str = repr(exp_scheduler)
        assert "decay_factor=0.9" in repr_str

    def test_default_decay_factor(self):
        model = torch.nn.Linear(4, 4)
        sche = ExponentDecayLr(
            optim_type="Adam",
            optim_conf={"lr": 1e-3},
            model=model,
            use_amp=False,
        )
        assert sche.decay_factor == 0.999
