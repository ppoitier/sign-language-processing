from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from slp.core.registry import BACKBONE_REGISTRY


class TemporalConvBlock(nn.Sequential):
    """Conv1d(kernel=5, no padding) - BN - ReLU."""

    def __init__(
            self,
            c_in: int,
            c_out: int,
            kernel_size: int = 5,
    ):
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )


@BACKBONE_REGISTRY.register("temporal_conv1d")
class Conv1DTemporalLayer(nn.Module):
    def __init__(
            self,
            in_dim: int = 512,
            hidden_dim: int = 1024,
            out_dim: int = 1024,
    ):
        super().__init__()
        self.net = nn.Sequential(
            TemporalConvBlock(in_dim, hidden_dim),  # T -> T - 4
            nn.MaxPool1d(kernel_size=2, stride=2),  #   -> (T - 4) // 2
            TemporalConvBlock(hidden_dim, out_dim), #   -> ... - 4
            nn.MaxPool1d(kernel_size=2, stride=2),  #   -> ... // 2
        )

    @staticmethod
    def output_lengths(lengths: Tensor) -> Tensor:
        """Map input frame counts to output time steps (needed for CTC)."""
        lengths = lengths - 4
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        lengths = lengths - 4
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        return lengths.clamp(min=0)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: (B, C, T) framewise features (e.g. from your pose backbone).
            mask: (B, ...) valid frames per sample, or None if unpadded.

        Returns:
            visual features (B, C_out, T').
        """
        return self.net(x)
