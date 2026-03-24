from typing import Any

from torch import nn, Tensor


class MultiTaskLoss(nn.Module):
    """Aggregates losses from multiple task-specific criteria."""

    def __init__(self, criteria: dict[str, nn.Module]):
        super().__init__()
        self.criteria = nn.ModuleDict(criteria)

    def forward(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Tensor]:
        losses = {}

        for task_name, criterion in self.criteria.items():
            if task_name in predictions:
                losses[task_name] = criterion(predictions[task_name], targets[task_name])

        losses["total_loss"] = sum(losses.values())
        return losses
