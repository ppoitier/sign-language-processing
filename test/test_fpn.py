import pytest
import torch

from slp.nn.necks.fpn import FeaturePyramidNetwork


def test_fpn_inference_1():
    N, C, T = 3, 64, 512

    x = [
        torch.randn(N, C, T),
        torch.randn(N, C, T // 2),
        torch.randn(N, C, T // 4),
        torch.randn(N, C, T // 8),
    ]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=4)
    y = model(x)

    for i in range(4):
        assert y[i].shape == (N, C, T // 2**i)
        assert torch.isfinite(y[i]).all()


def test_fpn_non_power_of_two_sizes():
    # Temporal sizes that don't divide cleanly
    N, C = 2, 32
    x = [
        torch.randn(N, C, 500),
        torch.randn(N, C, 250),
        torch.randn(N, C, 125),
        torch.randn(N, C, 62),
    ]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=4)
    y = model(x)

    for xi, yi in zip(x, y):
        assert yi.shape[-1] == xi.shape[-1]
        assert torch.isfinite(yi).all()


def test_fpn_single_level():
    N, C, T = 2, 16, 64
    x = [torch.randn(N, C, T)]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=1)
    y = model(x)

    assert len(y) == 1
    assert y[0].shape == (N, C, T)


def test_fpn_batch_size_one():
    N, C, T = 1, 32, 128
    x = [torch.randn(N, C, T // 2**i) for i in range(3)]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=3)
    y = model(x)

    for i, yi in enumerate(y):
        assert yi.shape == (N, C, T // 2**i)


def test_fpn_channel_projection():
    # c_in different from c_out
    N, C_in, C_out, T = 2, 64, 128, 256
    x = [torch.randn(N, C_in, T // 2**i) for i in range(4)]

    model = FeaturePyramidNetwork(c_in=C_in, c_out=C_out, n_levels=4)
    y = model(x)

    for i, yi in enumerate(y):
        assert yi.shape == (N, C_out, T // 2**i)


def test_fpn_heterogeneous_c_in():
    # Realistic backbone profile with different channels per level
    N, T = 2, 256
    c_in = [64, 128, 256, 512]
    c_out = 128
    x = [torch.randn(N, c, T // 2**i) for i, c in enumerate(c_in)]

    model = FeaturePyramidNetwork(c_in=c_in, c_out=c_out, n_levels=4)
    y = model(x)

    for i, yi in enumerate(y):
        assert yi.shape == (N, c_out, T // 2**i)


def test_fpn_top_down_pathway_active():
    # Zero out all but the coarsest level. If the top-down pathway is wired
    # correctly, the finest output must still be non-zero because information
    # flows from the coarsest level all the way up.
    N, C, T = 2, 32, 128
    n_levels = 4

    x = [torch.zeros(N, C, T // 2**i) for i in range(n_levels)]
    x[-1] = torch.randn(N, C, T // 2 ** (n_levels - 1))

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=n_levels)
    model.eval()
    with torch.no_grad():
        y = model(x)

    for i, yi in enumerate(y):
        assert yi.abs().sum() > 0, f"Level {i} output is all zero"


def test_fpn_gradient_flow():
    N, C, T = 2, 32, 128
    x = [torch.randn(N, C, T // 2**i) for i in range(4)]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=4)
    y = model(x)

    loss = sum(yi.sum() for yi in y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_fpn_wrong_number_of_inputs_raises():
    model = FeaturePyramidNetwork(c_in=32, c_out=32, n_levels=4)
    x = [torch.randn(2, 32, 64 // 2**i) for i in range(3)]  # only 3, not 4

    with pytest.raises(ValueError):
        model(x)


def test_fpn_determinism():
    N, C, T = 2, 32, 128
    x = [torch.randn(N, C, T // 2**i) for i in range(4)]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=4)
    model.eval()

    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    for a, b in zip(y1, y2):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fpn_dtype(dtype):
    N, C, T = 2, 32, 128
    x = [torch.randn(N, C, T // 2**i, dtype=dtype) for i in range(4)]

    model = FeaturePyramidNetwork(c_in=C, c_out=C, n_levels=4).to(dtype)
    y = model(x)

    for yi in y:
        assert yi.dtype == dtype
        assert torch.isfinite(yi).all()