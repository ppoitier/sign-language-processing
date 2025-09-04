import torch
from torch import nn
from torch.nn.functional import pad, unfold


class TridentHead(nn.Module):
    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.in_channels = 2 + 2 * (n_bins + 1)
        # self.fc = nn.Linear(in_channels, 2 + 2*(n_bins + 1))
        self.unfold = nn.Unfold(kernel_size=n_bins + 1, stride=1)
        self.register_buffer('bin_range', torch.arange(start=0, end=n_bins + 1, step=1).reshape(1, 1, -1))

    def forward(self, x):
        """
        Args:
            x: tensor of shape (N, T, C_in)
        Returns:
            out: tensor of shape (N, T, 2)
        """
        x_starts, x_ends, x_center_starts, x_center_ends = torch.split(x, [1, 1, self.n_bins + 1, self.n_bins + 1], dim=-1)

        x_starts = pad(x_starts, (0, 0, self.n_bins, 0), mode='constant', value=0.0)
        x_starts = x_starts.flatten(start_dim=1).unfold(dimension=1, size=self.n_bins + 1, step=1)
        start_probs = torch.softmax(x_starts + x_center_starts, dim=-1)
        d_start = torch.sum(self.bin_range * start_probs, dim=-1)

        x_ends = pad(x_ends, (0, 0, 0, self.n_bins), mode='constant', value=0.0)
        x_ends = x_ends.flatten(start_dim=1).unfold(dimension=1, size=self.n_bins + 1, step=1)
        end_probs = torch.softmax(x_ends + x_center_ends, dim=-1)
        d_end = torch.sum(self.bin_range * end_probs, dim=-1)

        return torch.stack([d_start, d_end], dim=-1)


class TridentModel(nn.Module):
    def __init__(self, backbone, n_trident_bins: int):
        super().__init__()
        self.backbone = backbone
        self.trident_head = TridentHead(n_trident_bins)
        self.n_trident_in_channels = 2 * (n_trident_bins + 1)

    def forward(self, x):
        x = self.backbone(x)
        out_channels = x.shape[-1]
        x_scores, x_trident = x.split([out_channels - self.n_trident_in_channels, self.n_trident_in_channels], dim=-1)
        x_trident = self.trident_head(x_trident)
        return torch.stack([x_scores, x_trident], dim=-1)


if __name__ == '__main__':
    _model = TridentHead(16)
    _input = torch.randn(3, 128, 2 + 2*17)
    _output = _model(_input)
    print(_output.shape)
