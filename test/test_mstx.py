import torch

from slp.nn.backbones.transformer import MultiStageTransformer
from slp.nn.necks.fpn import FeaturePyramidNetwork


def test_mstx_inference():
    model = MultiStageTransformer(
        in_channels=130,
        hidden_channels=128,
        max_length=1024,
        n_heads=4,
        n_stem_layers=2,
        n_branch_layers=4,
    )

    fpn = FeaturePyramidNetwork(c_in=128, c_out=128, n_levels=4)

    with torch.no_grad():
        x = torch.randn(1, 130, 1024)
        mask = torch.ones(1, 1, 1024).bool()
        y = model(x, mask)
        print([yy.shape for yy in y])
        y = fpn(y)
        print([yy.shape for yy in y])
