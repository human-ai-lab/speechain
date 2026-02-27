import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.abs import Criterion


class TestCriterion:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            Criterion()

    def test_subclass_must_implement_call(self):
        class IncompleteCriterion(Criterion):
            pass

        with pytest.raises(TypeError):
            IncompleteCriterion()

    def test_subclass_with_call_can_be_instantiated(self):
        class ConcreteCriterion(Criterion):
            def __call__(self, x):
                return x

        criterion = ConcreteCriterion()
        assert criterion is not None

    def test_criterion_init_is_optional(self):
        class ConcreteCriterion(Criterion):
            def __call__(self, x):
                return x

        criterion = ConcreteCriterion(some_param=42)
        assert criterion is not None

    def test_custom_criterion_init_is_called(self):
        class ConcreteCriterion(Criterion):
            def criterion_init(self, value=0):
                self.value = value

            def __call__(self, x):
                return x + self.value

        criterion = ConcreteCriterion(value=10)
        assert criterion.value == 10
