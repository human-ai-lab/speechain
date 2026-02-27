import pytest

torch = pytest.importorskip("torch")

from speechain.module.abs import Module


class TestModule:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            Module()

    def test_subclass_must_implement_module_init_and_forward(self):
        class IncompleteModule(Module):
            pass

        with pytest.raises(TypeError):
            IncompleteModule()

    def test_concrete_subclass_instantiation(self):
        class ConcreteModule(Module):
            def module_init(self, out_features: int = 4):
                self.linear = torch.nn.Linear(2, out_features)
                self.output_size = out_features

            def forward(self, x):
                return self.linear(x)

        module = ConcreteModule(out_features=4)
        assert module is not None
        assert module.output_size == 4

    def test_input_size_propagation(self):
        class ConcreteModule(Module):
            def module_init(self):
                self.output_size = self.input_size

            def forward(self, x):
                return x

        module = ConcreteModule(input_size=16)
        assert module.input_size == 16
        assert module.output_size == 16

    def test_forward_pass(self):
        class ConcreteModule(Module):
            def module_init(self, features: int = 4):
                self.linear = torch.nn.Linear(features, features)
                self.output_size = features

            def forward(self, x):
                return self.linear(x)

        module = ConcreteModule(features=4)
        x = torch.randn(2, 4)
        out = module(x)
        assert out.shape == (2, 4)
