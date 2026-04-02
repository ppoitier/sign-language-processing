import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slp.core.registry import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("rnn")
class RNN(nn.Module):
    """
    A Recurrent Neural Network backbone. The different types of supported RNNs are 'lstm', 'gru', and 'rnn'.

    Shape Notation:
        - N: Batch size.
        - C_in: Number of input channels.
        - C_hidden: Number of hidden channels for the RNN layers.
        - C_out: Number of output channels (usually num_classes).
        - T: Temporal sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        rnn_type: str = "lstm",
    ):
        super().__init__()

        rnn_type = rnn_type.lower()
        rnn_type = rnn_type.lower()
        rnn_classes = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}

        if rnn_type not in rnn_classes:
            raise ValueError(
                f"rnn_type must be one of {list(rnn_classes.keys())}, got '{rnn_type}'"
            )

        self.rnn = rnn_classes[rnn_type](
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        rnn_out_channels = hidden_channels * 2 if bidirectional else hidden_channels
        self.conv_out = nn.Conv1d(rnn_out_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            Logits tensor of shape (N, C_out, T).
        """
        # PyTorch RNNs expect (N, T, C_in) when batch_first=True
        x = x.permute(0, 2, 1).contiguous()

        # 1. Compute valid sequence lengths from the mask
        lengths = mask.squeeze(1).sum(dim=1).to(torch.int64).cpu()

        # 2. Pack the sequence
        packed_x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # 3. Pass through RNN
        packed_out, _ = self.rnn(packed_x)

        # 4. Unpack back to a padded tensor of shape (N, T, rnn_out_channels)
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        # Permute back to (N, C, T) for the 1D convolution and final output
        out = out.permute(0, 2, 1)

        logits = self.conv_out(out)

        return logits * mask
