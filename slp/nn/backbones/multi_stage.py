from typing import Iterable

from torch import nn, Tensor
from torch.nn import functional as F


class MultiStageBackbone(nn.Module):
    """
    A sequential multi-stage backbone for hierarchical feature extraction.

    This module passes input sequentially through a series of stages. It is
    designed for spatiotemporal data where stages may reduce the temporal
    dimension (e.g., via strided convolutions or pooling).
    """

    def __init__(self, stages: Iterable[nn.Module]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]:
        """
        Args:
            x: Input tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            outs: A list of intermediate feature tensors from each stage.
        """
        outs: list[Tensor] = []

        for stage in self.stages:
            x = stage(x, mask)
            outs.append(x)

            # Fallback: Downsample the mask if needed. Typically handled by the stages.
            if mask.size(-1) != x.size(-1):
                mask = F.interpolate(
                    mask.float(), size=x.size(-1), mode="nearest"
                ).bool()

        return outs


class IterativeRefinementModel(nn.Module):
    """
    An iterative refinement model for progressive prediction smoothing.

    The architecture is explicitly split into two phases:
    1. An initial stage that maps raw input features (C_in) to logits (C_classes).
    2. A sequence of refinement stages that map bounded predictions back to logits
       (C_classes -> C_classes).

    An activation function is applied to the predictions between stages to convert
    raw logits into bounded values (e.g., probabilities) before feeding them into
    the subsequent refinement stage.

    Args:
        initial_stage: Module computing the base predictions from feature inputs.
        refinement_stages: Iterable of modules, each refining the previous stage's
                           activated output.
        activation: Function to bound logits between stages. Defaults to nn.Identity().
    """

    def __init__(
        self,
        initial_stage: nn.Module,
        refinement_stages: Iterable[nn.Module],
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.initial_stage = initial_stage
        self.refinement_stages = nn.ModuleList(refinement_stages)
        self.activation = nn.Identity() if activation is None else activation

    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]:
        """
        Args:
            x: Input tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            outs: A list of logit tensors from each stage, typically
                  used to compute the loss at each refinement step.
        """
        outs: list[Tensor] = []

        # Initial prediction from the features
        out = self.initial_stage(x, mask)
        outs.append(out)

        # Iterative refinement of the predictions
        for stage in self.refinement_stages:
            out = stage(self.activation(out), mask)
            outs.append(out)

            # Fallback: Downsample the mask if needed. Typically handled by the stages.
            if mask.size(-1) != x.size(-1):
                mask = F.interpolate(
                    mask.float(), size=x.size(-1), mode="nearest"
                ).bool()

        return outs
