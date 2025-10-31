import torch
from torch import nn, Tensor
from typing import overload, Literal


class WeightedSumLoss(nn.Module):
    """
    Calculates a weighted sum of multiple loss functions.
    """

    def __init__(
        self,
        losses: list[nn.Module],
        weights: list[float],
    ):
        """
        Args:
            losses (list[nn.Module]): A list of loss function modules.
            weights (list[float]): A list of weights corresponding to the loss functions.
        """
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError(
                "The `losses` and `weights` lists must have the same length."
            )

        self.losses = nn.ModuleList(losses)
        self.register_buffer('weights', torch.tensor(weights))

    @overload
    def forward(
        self, logits: Tensor, targets: Tensor, *, return_sub_components: Literal[True]
    ) -> tuple[Tensor, list[Tensor]]: ...

    @overload
    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        return_sub_components: Literal[False] = False,
    ) -> Tensor: ...

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        return_sub_components: bool | None = None,
    ) -> tuple[Tensor, list[Tensor]] | Tensor:
        """
        Computes the total weighted loss.

        Args:
            logits (Tensor): The model's predictions, typically of shape (N, C_in, ...).
            targets (Tensor): The ground truth labels, typically of shape (N, ...).
            return_sub_components (bool | None): Overrides the instance-level setting if provided.

        Returns:
            - If `return_sub_components` is True: A tuple containing the total weighted loss (scalar tensor)
              and a list of the individual unweighted (detached) losses.
            - If `return_sub_components` is False: The total weighted loss (scalar tensor).
        """
        individual_losses = [loss_fn(logits, targets) for loss_fn in self.losses]
        total_loss = torch.dot(self.weights, torch.stack(individual_losses))
        if return_sub_components:
            detached_losses = [loss.detach() for loss in individual_losses]
            return total_loss, detached_losses
        return total_loss
