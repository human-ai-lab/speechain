"""Tests for speechain/utilbox/train_util.py"""

import pytest

torch = pytest.importorskip("torch")

from speechain.utilbox.train_util import (
    float_near_round,
    get_min_indices_by_freq,
    get_padding_by_dilation,
    make_len_from_mask,
    make_mask_from_len,
)


class TestMakeMaskFromLen:
    def test_basic_3d(self):
        lengths = torch.tensor([3, 5, 2])
        mask = make_mask_from_len(lengths)
        assert mask.shape == (3, 1, 5)
        # first sequence: first 3 positions True, rest False
        assert mask[0, 0, :3].all()
        assert not mask[0, 0, 3:].any()

    def test_basic_2d(self):
        lengths = torch.tensor([2, 4])
        mask = make_mask_from_len(lengths, return_3d=False)
        assert mask.shape == (2, 4)
        assert mask[0, :2].all()
        assert not mask[0, 2:].any()

    def test_explicit_max_len(self):
        lengths = torch.tensor([3])
        mask = make_mask_from_len(lengths, max_len=10)
        assert mask.shape == (1, 1, 10)

    def test_max_len_too_small_raises(self):
        lengths = torch.tensor([5])
        with pytest.raises(AssertionError):
            make_mask_from_len(lengths, max_len=3)

    def test_bool_dtype(self):
        lengths = torch.tensor([2, 3])
        mask = make_mask_from_len(lengths, mask_type=torch.bool)
        assert mask.dtype == torch.bool

    def test_float_dtype(self):
        lengths = torch.tensor([2])
        mask = make_mask_from_len(lengths, mask_type=torch.float32)
        assert mask.dtype == torch.float32


class TestMakeLenFromMask:
    def test_3d_mask(self):
        mask = torch.zeros(2, 1, 5, dtype=torch.bool)
        mask[0, 0, :3] = True
        mask[1, 0, :4] = True
        lengths = make_len_from_mask(mask)
        assert lengths.tolist() == [3, 4]

    def test_2d_mask(self):
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :5] = True
        lengths = make_len_from_mask(mask)
        assert lengths.tolist() == [2, 5]

    def test_roundtrip(self):
        original = torch.tensor([3, 5, 2])
        mask = make_mask_from_len(original)
        recovered = make_len_from_mask(mask)
        assert recovered.tolist() == original.tolist()


class TestFloatNearRound:
    def test_below_half_rounds_down(self):
        assert float_near_round(2.3) == 2

    def test_exactly_half_rounds_up(self):
        assert float_near_round(2.5) == 3

    def test_above_half_rounds_up(self):
        assert float_near_round(2.7) == 3

    def test_zero(self):
        assert float_near_round(0.0) == 0

    def test_whole_number(self):
        assert float_near_round(4.0) == 4

    def test_negative_like(self):
        # 1.49 -> rounds to 1
        assert float_near_round(1.49) == 1


class TestGetPaddingByDilation:
    def test_kernel3_dilation1(self):
        # (3*1 - 1) / 2 = 1
        assert get_padding_by_dilation(3, 1) == 1

    def test_kernel5_dilation1(self):
        # (5*1 - 1) / 2 = 2
        assert get_padding_by_dilation(5, 1) == 2

    def test_kernel3_dilation2(self):
        # (3*2 - 2) / 2 = 2
        assert get_padding_by_dilation(3, 2) == 2

    def test_kernel1_dilation1(self):
        assert get_padding_by_dilation(1, 1) == 0

    def test_default_dilation(self):
        # dilation defaults to 1
        assert get_padding_by_dilation(3) == 1


class TestGetMinIndicesByFreq:
    def test_returns_minimum(self):
        freq = {"a": 5, "b": 1, "c": 3}
        indices, updated = get_min_indices_by_freq(freq, shuffle=False)
        assert indices == ["b"]
        assert updated["b"] == 2  # incremented by 1

    def test_multiple_chosen(self):
        freq = {"a": 10, "b": 1, "c": 2}
        indices, updated = get_min_indices_by_freq(
            freq, shuffle=False, chosen_idx_num=2
        )
        assert len(indices) == 2
        assert "b" in indices

    def test_freq_weights_applied(self):
        freq = {"a": 0, "b": 10}
        indices, updated = get_min_indices_by_freq(
            freq, shuffle=False, chosen_idx_num=1, freq_weights=[5]
        )
        assert indices == ["a"]
        assert updated["a"] == 5

    def test_returns_updated_dict(self):
        freq = {"x": 0, "y": 1}
        _, updated = get_min_indices_by_freq(freq, shuffle=False)
        assert updated["x"] == 1  # was 0, now incremented by 1

    def test_weight_length_mismatch_raises(self):
        freq = {"a": 0, "b": 1}
        with pytest.raises(AssertionError):
            get_min_indices_by_freq(freq, chosen_idx_num=2, freq_weights=[1])
