import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels: int, max_length: int):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, in_channels, 2) * (-math.log(10000.0) / in_channels)).unsqueeze(0)
        pe = torch.zeros(1, max_length, in_channels)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, L, C_in)
        """
        return x + self.pe


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N, C_in, L = 1, 320, 256
    pose_encoder = PositionalEncoding(in_channels=C_in, max_length=L)
    example_x = torch.zeros(N, L, C_in)
    example_y = pose_encoder(example_x)

    plt.figure()
    plt.title('Example of positional encoding $(C_{in}=%s, L=%s)$' % (C_in, L))
    plt.xlabel('sequence length')
    plt.ylabel('# channels')
    plt.imshow(example_y.squeeze().detach().numpy().T)
    plt.tight_layout()
    plt.show()
