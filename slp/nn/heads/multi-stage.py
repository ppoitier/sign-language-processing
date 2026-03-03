from typing import Iterable

from torch import nn, Tensor


class SharedMultiStageHead(nn.Module):
    """
    A multi-stage head wrapper where weights are shared across all stages.

    The exact same head model instance is applied sequentially to every feature
    tensor in the input list.

    Shape Notation:
        - N: Batch size.
        - C_in: Number of input channels (feature dimension).
        - C_out: Number of output channels (e.g., number of sign classes or boundary logits).
        - T_l: Temporal sequence length at stage l. This can vary across stages
               if the backbone reduces temporal resolution.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: The head model to apply to each feature tensor.
        """
        super().__init__()
        self.model = model

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Args:
            features: A list of input tensors of shape (N, C_in, T_l).

        Returns:
            A list of output tensors of shape (N, C_out, T_l).
        """
        return [self.model(z) for z in features]


class IndependentMultiStageHead(nn.Module):
    """
    A multi-stage head wrapper where each stage has its own independent weights.

    A distinct head model is applied to the corresponding feature tensor in the
    input list.

    Shape Notation:
        - N: Batch size.
        - C_in: Number of input channels (feature dimension).
        - C_out: Number of output channels (e.g., number of sign classes or boundary logits).
        - T_l: Temporal sequence length at stage l. This can vary across stages
               if the backbone reduces temporal resolution.
    """

    def __init__(self, models: Iterable[nn.Module]):
        """
        Args:
            models: An iterable of distinct head models, one for each stage
                    of the input features.
        """
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Args:
            features: A list of input tensors of shape (N, C_in, T_l).

        Returns:
            A list of output tensors of shape (N, C_out, T_l).
        """
        if len(features) != len(self.models):
            raise ValueError(
                f"Number of input features ({len(features)}) does not match "
                f"the number of independent head models ({len(self.models)})."
            )

        return [model(z) for model, z in zip(self.models, features)]