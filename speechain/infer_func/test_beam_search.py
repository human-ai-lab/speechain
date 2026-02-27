import pytest

torch = pytest.importorskip("torch")

from speechain.infer_func.beam_search import BeamHypotheses


class TestBeamHypotheses:
    def test_instantiation(self):
        bh = BeamHypotheses(beam_size=4, max_length=50, length_penalty=1.0)
        assert len(bh) == 0

    def test_add_hypothesis(self):
        bh = BeamHypotheses(beam_size=4, max_length=50, length_penalty=1.0)
        hyp = torch.tensor([1, 2, 3])
        bh.add(hyp, -1.5)
        assert len(bh) == 1

    def test_beam_size_limit(self):
        bh = BeamHypotheses(beam_size=2, max_length=50, length_penalty=1.0)
        for i in range(5):
            bh.add(torch.tensor([1, 2, 3]), float(-i - 1))
        assert len(bh) <= 2

    def test_is_done_false_when_not_full(self):
        bh = BeamHypotheses(beam_size=4, max_length=50, length_penalty=1.0)
        assert not bh.is_done(-1.0, curr_len=5)

    def test_best_hypothesis_retained(self):
        bh = BeamHypotheses(beam_size=2, max_length=50, length_penalty=1.0)
        bh.add(torch.tensor([1]), -0.1)
        bh.add(torch.tensor([2]), -0.5)
        bh.add(torch.tensor([3]), -10.0)
        scores = [s for s, _ in bh.beams]
        assert max(scores) >= -0.5


class TestBeamSearchImport:
    def test_module_importable(self):
        import speechain.infer_func.beam_search as bs

        assert hasattr(bs, "BeamHypotheses")
