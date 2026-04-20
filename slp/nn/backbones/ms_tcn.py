from torch import nn, Tensor

from slp.nn.blocks.tcn.dilated_layers import DilatedResidualLayer
from slp.nn.backbones.multi_stage import IterativeRefinementModel

from slp.core.registry import BACKBONE_REGISTRY


class SingleStageTCN(nn.Module):
    """
    A single stage of the Temporal Convolutional Network.

    Shape Notation:
        - N: Batch size.
        - C_in: Number of input channels.
        - C_hidden: Number of hidden channels for the dilated layers.
        - C_out: Number of output channels (usually num_classes).
        - T: Temporal sequence length.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        self.dilated_layers = nn.ModuleList(
            [
                DilatedResidualLayer(channels=hidden_channels, dilation=2**i)
                for i in range(n_layers)
            ]
        )

        self.conv_out = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            Logits tensor of shape (N, C_out, T).
        """
        out = self.conv_in(x)
        for layer in self.dilated_layers:
            out = layer(out, mask)
        out = self.conv_out(out)
        return out * mask


@BACKBONE_REGISTRY.register("ms-tcn")
class MultiStageTCN(nn.Module):
    """
    A Multi-Stage Temporal Convolutional Network.

    Utilizes the IterativeRefinementModel to sequentially refine predictions.
    The first stage maps input features to logits. Subsequent stages map
    activated probabilities back to refined logits.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            n_stages: int,
            n_layers: int,
            activation: str = 'softmax',
    ):
        super().__init__()

        initial_stage = SingleStageTCN(in_channels, hidden_channels, out_channels, n_layers)
        refinement_stages = [
            SingleStageTCN(out_channels, hidden_channels, out_channels, n_layers)
            for _ in range(n_stages - 1)
        ]

        act = None
        if activation == 'softmax':
            act = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            act = nn.Sigmoid()

        self.model = IterativeRefinementModel(initial_stage, refinement_stages, activation=act)

    def forward(self, x: Tensor, mask: Tensor) -> list[Tensor]:
        """
        Args:
            x: Input feature tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            A list containing N_stages tensors, each of shape (N, C_out, T).
        """
        return self.model(x, mask)
