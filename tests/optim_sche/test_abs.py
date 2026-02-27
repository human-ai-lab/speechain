import pytest

torch = pytest.importorskip("torch")

from speechain.optim_sche.abs import OptimScheduler


class TestOptimScheduler:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            OptimScheduler(
                optim_type="Adam",
                optim_conf={"lr": 1e-3},
                model=torch.nn.Linear(2, 2),
            )

    def test_subclass_must_implement_sche_init_and_update_lr(self):
        class IncompleteScheduler(OptimScheduler):
            pass

        with pytest.raises(TypeError):
            IncompleteScheduler(
                optim_type="Adam",
                optim_conf={"lr": 1e-3},
                model=torch.nn.Linear(2, 2),
            )

    def test_concrete_subclass_get_lr(self):
        class SimpleScheduler(OptimScheduler):
            def sche_init(self):
                pass

            def update_lr(self, real_step, epoch_num):
                return 1e-3

        model = torch.nn.Linear(2, 2)
        scheduler = SimpleScheduler(
            optim_type="Adam",
            optim_conf={"lr": 1e-3},
            model=model,
            use_amp=False,
        )
        assert scheduler.get_lr() == pytest.approx(1e-3)

    def test_state_dict_roundtrip(self):
        class SimpleScheduler(OptimScheduler):
            def sche_init(self):
                pass

            def update_lr(self, real_step, epoch_num):
                return 1e-3

        model = torch.nn.Linear(2, 2)
        scheduler = SimpleScheduler(
            optim_type="Adam",
            optim_conf={"lr": 1e-3},
            model=model,
            use_amp=False,
        )
        state = scheduler.state_dict()
        assert "optimizer" in state
        assert "scaler" in state
