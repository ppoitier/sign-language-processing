from typing import Iterable

from torch import nn, Tensor
from torch.nn import functional as F


class MultiStageModel(nn.Module):
    def __init__(
        self,
        stages: Iterable[nn.Module],
        activation: None | nn.Module = None,
    ):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.activation = nn.Identity() if activation is None else activation

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: tensor of shape (N, C_in, T)
            mask: tensor of shape (N, 1, T)

        Returns:
            logits: tuple containing N_layers tensors (N, C_in, T)
        """
        out = x
        outs: tuple[Tensor] = tuple()
        for idx, stage in enumerate(self.stages):
            out = stage(self.activation(out) if idx > 0 else out, mask)
            outs += (out,)
            if mask.size(-1) != out.size(-1):
                mask = F.interpolate(mask.float(), out.shape[-1], mode="nearest").bool()
        return outs
