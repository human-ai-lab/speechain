import pytest

torch = pytest.importorskip("torch")

from speechain.optim_sche.noam import Noamlr


@pytest.fixture
def noam_scheduler():
    model = torch.nn.Linear(4, 4)
    return Noamlr(
        optim_type="Adam",
        optim_conf={"lr": 1e-3},
        model=model,
        warmup_steps=4000,
        use_amp=False,
    )


class TestNoamlr:
    def test_instantiation(self, noam_scheduler):
        assert noam_scheduler is not None
        assert noam_scheduler.warmup_steps == 4000

    def test_warmup_phase_increases(self, noam_scheduler):
        lr_step1 = noam_scheduler.update_lr(1, 0)
        lr_step4000 = noam_scheduler.update_lr(4000, 0)
        assert lr_step1 < lr_step4000

    def test_decay_phase_decreases(self, noam_scheduler):
        lr_step4000 = noam_scheduler.update_lr(4000, 0)
        lr_step8000 = noam_scheduler.update_lr(8000, 0)
        assert lr_step8000 < lr_step4000

    def test_update_lr_positive(self, noam_scheduler):
        lr = noam_scheduler.update_lr(100, 0)
        assert lr > 0

    def test_extra_repr(self, noam_scheduler):
        repr_str = repr(noam_scheduler)
        assert "warmup_steps=4000" in repr_str

    def test_d_model_formula(self):
        model = torch.nn.Linear(4, 4)
        sche = Noamlr(
            optim_type="Adam",
            optim_conf={"lr": 1e-3},
            model=model,
            d_model=256,
            warmup_steps=4000,
            use_amp=False,
        )
        assert sche.d_model == 256
        lr_peak = sche.update_lr(4000, 0)
        lr_decay = sche.update_lr(8000, 0)
        assert lr_decay < lr_peak
