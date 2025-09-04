import torch
from torch import Tensor
import torch.nn.functional as F


def masked_mse_loss(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if input.shape != target.shape:
        raise ValueError("Input and target must have the same shape.")
    if mask.ndim != target.ndim:
        raise ValueError("Mask and target must have the same number of dimensions.")
    mask = mask.expand_as(input).bool()
    loss_none = F.mse_loss(input, target, reduction='none')
    selected = loss_none[mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=input.device)
    return selected.mean()


def temporal_mse_loss(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    # Expand mask to match feature dimensions (e.g., (B, T) -> (B, T, C))
    mask_expanded = mask.view(*mask.shape, *(1,) * (input.dim() - mask.dim()))
    return masked_mse_loss(input, target, mask_expanded)
