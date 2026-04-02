"""
Code inspired from SlowFast (https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/common.py).
"""
import torch
from torch import nn, Tensor, rand


def drop_path(x: Tensor, drop_p: float = 0.0, training: bool = True) -> Tensor:
    if drop_p == 0.0 or not training:
        return x
    keep_p = 1 - drop_p

    # Randomly create a mask of shape (batch_size, 1, 1, ..., 1) where
    # each value has a probability `drop_p` of being 0
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_p + rand(mask_shape, dtype=x.dtype, device=x.device)
    mask.floor_()

    # Apply the mask and scale the remaining activations to
    # compensate the average reduction in activation.
    output = x.div(keep_p) * mask
    return output


class DropPath(nn.Module):
    def __init__(self, drop_p: float):
        """
        DropPath (or Stochastic Depth) randomly drops certain paths in deep neural networks during training.
        It is a regularization technique particularly used in the context of deep convolutional and transformer models.
        It's conceptually similar to dropout, but with paths and not nodes.
        """
        super().__init__()
        self.drop_p = drop_p

    def forward(self, x):
        return drop_path(x, self.drop_p, self.training)


class ChannelWiseScale(nn.Module):
    def __init__(self, n_channels: int, init_scale_value: float = 1e-4):
        """
        ChannelWiseScale multiply a trainable scaling factor for each channel.

        Args:
            n_channels: Number of input channels
            init_scale_value: Initial value for the scaling factors (Default=1e-4)

        Shape:
            - Input of shape (B, C, ...) where B is the batch size, and C the number of channels.
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((1, n_channels, 1), init_scale_value, dtype=torch.float)
        )

    def forward(self, x):
        return self.scale * x


class AffineDropPath(nn.Module):
    def __init__(self, n_channels: int, drop_p: float = 0.0, init_scaling_value: float = 1e-4):
        super().__init__()
        self.affine_drop = nn.Sequential(
            ChannelWiseScale(n_channels, init_scaling_value),
            DropPath(drop_p),
        )

    def forward(self, x):
        return self.affine_drop(x)
