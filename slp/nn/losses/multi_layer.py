from torch import nn, Tensor
from typing import Any


class MultiLayerLoss(nn.Module):
    """Applies a base criterion across a list of multi-stage predictions."""

    def __init__(self, base_criterion: nn.Module, apply_on_all_stages: bool = True):
        super().__init__()
        self.base_criterion = base_criterion
        self.apply_on_all_stages = apply_on_all_stages

    def forward(self, predictions: list[Tensor], targets: dict[str, Any]) -> Tensor:
        stages = predictions if self.apply_on_all_stages else [predictions[-1]]

        # Compute loss for each active stage and average them
        total_loss = sum(
            self.base_criterion(stage_pred, targets) for stage_pred in stages
        )
        return total_loss / len(stages)
