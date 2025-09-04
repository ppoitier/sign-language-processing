from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, H, W)
        out = self.layers(x)
        return out + self.residual(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, hidden_channels=(32, 64, 128)):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1)]
        for i in range(1, len(hidden_channels)):
            layers += [
                ResidualBlock(hidden_channels[i - 1], hidden_channels[i]),
                nn.Conv2d(hidden_channels[i], hidden_channels[i], kernel_size=3, stride=2, padding=1),
            ]
        layers += [
            nn.GroupNorm(32, hidden_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, H, W) -> (N, C_out, H', W')
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_channels=(128, 64, 32)):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1)]
        for i in range(1, len(hidden_channels)):
            layers += [
                ResidualBlock(hidden_channels[i - 1], hidden_channels[i]),
                nn.Upsample(scale_factor=2, mode="nearest"),
                ResidualBlock(hidden_channels[i], hidden_channels[i]),
            ]
        layers += [
            nn.GroupNorm(32, hidden_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
