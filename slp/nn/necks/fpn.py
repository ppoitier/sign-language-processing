from torch import nn, Tensor
from torch.nn import functional as F

from slp.core.registry import NECK_REGISTRY
from slp.nn.blocks.norm.layer_norm import LayerNorm


@NECK_REGISTRY.register("fpn")
class FeaturePyramidNetwork(nn.Module):
    """
    1D Feature Pyramid Network (FPN) for temporal sequence features.

    This module fuses a pyramid of multi-scale temporal features extracted from a
    backbone. It applies 1x1 lateral convolutions to standardize the channel dimensions,
    followed by a top-down pathway that upsamples and adds coarser features to finer
    features. Finally, a 3x3 smoothing convolution is applied to each fused level
    to mitigate upsampling aliasing.

    Args:
        c_in: Number of input channels from the backbone features.
        c_out: Number of output channels for the FPN features.
        n_levels: Number of feature levels in the input pyramid.
    """
    def __init__(
            self,
            c_in: int | list[int],
            c_out: int,
            n_levels: int,
    ):
        super().__init__()
        self.n_levels = n_levels

        if isinstance(c_in, int):
            c_in = [c_in] * n_levels
        elif len(c_in) != n_levels:
            raise ValueError(f"Expected c_in to have {n_levels} elements, got {len(c_in)}.")

        # 1x1 convs to unify channel dimensions across all backbone levels
        self.lateral_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=c_in[i],
                out_channels=c_out,
                kernel_size=1,
                bias=False,
            ) for i in range(n_levels)
        ])

        # 3x3 smoothing convs applied after the top-down addition
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=c_out,
                    out_channels=c_out,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    groups=c_out,  # Note: depthwise convolution to save parameters
                ),
                LayerNorm(c_out),
            ) for _ in range(n_levels)
        ])

    def forward(self, embeddings: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Args:
            embeddings: Tuple of Tensors of shape (N, C, T_i), ordered from
                        finest temporal resolution to coarsest.

        Returns:
            Tuple of Tensors of shape (N, out_channels, T_i) representing the
            fused FPN features, maintaining the same order as the input.
        """
        if len(embeddings) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} embeddings, got {len(embeddings)}.")

        # 1. Lateral connections
        laterals = [
            conv(x) for conv, x in zip(self.lateral_convs, embeddings)
        ]

        # 2. Top-down pathway
        for i in range(self.n_levels - 1, 0, -1):
            # Target the exact temporal dimension of the finer level
            target_size = laterals[i - 1].size(-1)

            upsampled = F.interpolate(
                laterals[i],
                size=target_size,
                mode="nearest",
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3. Smoothing convolutions
        fpn_feats = tuple(
            conv(x) for conv, x in zip(self.fpn_convs, laterals)
        )

        return fpn_feats
