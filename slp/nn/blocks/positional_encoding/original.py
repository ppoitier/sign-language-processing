import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels: int, max_length: int):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, in_channels, 2) * (-math.log(10000.0) / in_channels)).unsqueeze(0)
        pe = torch.zeros(1, max_length, in_channels)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, L, C_in)
        """
        return x + self.pe[:, : x.size(2)]


if __name__ == "__main__":
    N, C_in, L = 1, 320, 1024
    pose_encoder = PositionalEncoding(in_channels=C_in, max_length=L)

    example_x = torch.zeros(N, C_in, L)
    example_y = pose_encoder(example_x)

    plt.figure(figsize=(6, 4))
    plt.title(f"Positional Encoding ($C_{{in}}={C_in}, L={L}$)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Channels")

    plt.imshow(
        example_y.squeeze().detach().numpy(), aspect="auto", interpolation="none"
    )

    plt.tight_layout()
    plt.show()
