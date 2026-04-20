import pytest
import torch
from torch import Tensor

from slp.nn.backbones.transformer import MultiStageTransformer
from slp.nn.necks.fpn import FeaturePyramidNetwork


@pytest.fixture
def default_cfg() -> dict:
    return dict(
        in_channels=32,
        hidden_channels=64,
        max_length=128,
        n_heads=4,
        n_stem_layers=2,
        n_branch_layers=4,
    )


@pytest.fixture
def model(default_cfg) -> MultiStageTransformer:
    return MultiStageTransformer(**default_cfg)


def make_batch(
    batch_size: int = 1,
    in_channels: int = 32,
    length: int = 128,
    pad_last: int = 0,
) -> tuple[Tensor, Tensor]:
    """Return (x, mask) with optional right-side padding."""
    x = torch.randn(batch_size, in_channels, length)
    mask = torch.ones(batch_size, 1, length, dtype=torch.bool)
    if pad_last > 0:
        mask[..., -pad_last:] = 0
    return x, mask


class TestMultiStageTransformer:

    def test_output_is_list_of_tensors(self, model, default_cfg):
        x, mask = make_batch()
        with torch.no_grad():
            y = model(x, mask)
        assert isinstance(y, list)
        assert all(isinstance(t, Tensor) for t in y)

    def test_output_length_equals_n_branch_layers_plus_one(self, model, default_cfg):
        """stage0 (full-res) + one tensor per branch layer."""
        x, mask = make_batch()
        with torch.no_grad():
            y = model(x, mask)
        assert len(y) == default_cfg["n_branch_layers"] + 1

    def test_stage0_is_full_resolution(self, model, default_cfg):
        T = default_cfg["max_length"]
        C = default_cfg["hidden_channels"]
        x, mask = make_batch(length=T)
        with torch.no_grad():
            y = model(x, mask)
        assert y[0].shape == (1, C, T)

    def test_pyramid_shapes(self, model, default_cfg):
        """Each branch stage halves the temporal dimension."""
        T = default_cfg["max_length"]
        C = default_cfg["hidden_channels"]
        x, mask = make_batch(length=T)
        with torch.no_grad():
            y = model(x, mask)
        for i in range(default_cfg["n_branch_layers"]):
            expected = (1, C, T // 2 ** (i + 1))
            assert y[i + 1].shape == expected, (
                f"Stage {i + 1}: expected {expected}, got {y[i + 1].shape}"
            )

    def test_batch_size_propagated(self, model, default_cfg):
        N = 3
        x, mask = make_batch(batch_size=N)
        with torch.no_grad():
            y = model(x, mask)
        assert all(t.shape[0] == N for t in y)

    def test_no_nan_in_output(self, model, default_cfg):
        x, mask = make_batch()
        with torch.no_grad():
            y = model(x, mask)
        assert all(not t.isnan().any() for t in y)

    # def test_padded_mask_does_not_affect_valid_region(self, model, default_cfg):
    #     """Outputs for valid frames should be identical regardless of what
    #     is in the padded region."""
    #     T = default_cfg["max_length"]
    #     PAD = T // 4
    #
    #     x, mask_full = make_batch(length=T)
    #     _, mask_padded = make_batch(length=T, pad_last=PAD)
    #
    #     with torch.no_grad():
    #         y_full = model(x, mask_full)
    #         y_padded = model(x, mask_padded)
    #
    #     valid_len = T - PAD
    #     assert torch.allclose(
    #         y_full[0][..., :valid_len],
    #         y_padded[0][..., :valid_len],
    #         atol=1e-5,
    #     ), "Padding leaked into the valid region at full resolution."

    @pytest.mark.parametrize("n_branch_layers", [1, 2, 6])
    def test_variable_n_branch_layers(self, default_cfg, n_branch_layers):
        cfg = {**default_cfg, "n_branch_layers": n_branch_layers}
        model = MultiStageTransformer(**cfg)
        x, mask = make_batch()
        with torch.no_grad():
            y = model(x, mask)
        assert len(y) == n_branch_layers + 1

    @pytest.mark.parametrize("pos_encoding", ["rope", "sinusoidal", None])
    def test_pos_encoding_variants(self, default_cfg, pos_encoding):
        model = MultiStageTransformer(**default_cfg, pos_encoding=pos_encoding)
        x, mask = make_batch()
        with torch.no_grad():
            y = model(x, mask)
        assert y[0].shape[1] == default_cfg["hidden_channels"]


class TestMultiStageTransformerWithFPN:

    def _run(self, model, fpn, length):
        x, mask = make_batch(length=length)
        with torch.no_grad():
            return fpn(model(x, mask))

    def test_output_length_equals_n_branch_layers_plus_one(self, model, default_cfg):
        x, mask = make_batch()
        with torch.no_grad():
            y = model(x, mask)
        assert len(y) == default_cfg["n_branch_layers"] + 1

    def test_fpn_all_levels_shapes(self, model, default_cfg):
        n_levels = default_cfg["n_branch_layers"] + 1  # was: n_branch_layers
        T = default_cfg["max_length"]
        C = default_cfg["hidden_channels"]
        fpn = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=n_levels)
        y = self._run(model, fpn, T)
        assert isinstance(y, tuple) and len(y) == n_levels
        # level 0 is now full-res, rest are downsampled
        assert y[0].shape == (1, C, T)
        for i in range(1, n_levels):
            assert y[i].shape == (1, C, T // 2**i)

    def test_fpn_finest_returns_tensor_not_list(self, model, default_cfg):
        C = default_cfg["hidden_channels"]
        fpn = FeaturePyramidNetwork(
            c_in=C,
            c_out=C,
            n_levels=default_cfg["n_branch_layers"] + 1,  # was: n_branch_layers
            output_mode="finest",
        )
        y = self._run(model, fpn, default_cfg["max_length"])
        assert isinstance(y, Tensor)

    def test_fpn_finest_is_full_resolution(self, model, default_cfg):
        T = default_cfg["max_length"]
        C = default_cfg["hidden_channels"]
        fpn = FeaturePyramidNetwork(
            c_in=C,
            c_out=C,
            n_levels=default_cfg["n_branch_layers"] + 1,  # was: n_branch_layers
            output_mode="finest",
        )
        y = self._run(model, fpn, T)
        assert y.shape == (1, C, T)

    def test_fpn_output_no_nan(self, model, default_cfg):
        C = default_cfg["hidden_channels"]
        fpn = FeaturePyramidNetwork(
            c_in=C,
            c_out=C,
            n_levels=default_cfg["n_branch_layers"]
            + 1,  # was: n_branch_layers (hardcoded 4)
        )
        y = self._run(model, fpn, default_cfg["max_length"])
        assert all(not t.isnan().any() for t in y)
