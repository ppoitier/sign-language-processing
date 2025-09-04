from collections import OrderedDict

import torch
from torch import nn, Tensor

from slp.config.templates.model import ModelConfig, HeadConfig
from slp.nn.blocks.tcn.tcn import MultiStageTCN


class RefinementLayersSegmentationModel(nn.Module):
    def __init__(self, encoder_kwargs, heads_config: OrderedDict[str, HeadConfig]):
        super().__init__()
        self.encoder = MultiStageTCN(**encoder_kwargs)

        self.heads = nn.ModuleDict()
        self.head_splits = [config.in_channels for config in heads_config.values()]
        for name, config in heads_config.items():
            match config.layer:
                case 'identity':
                    self.heads[name] = nn.Identity()
                case 'linear':
                    self.heads[name] = nn.Conv1d(
                        in_channels=config.in_channels,
                        out_channels=config.out_channels,
                        kernel_size=1,
                    )

    def forward(self, x: Tensor, mask: Tensor) -> dict[str, tuple[Tensor, ...]]:
        """
        Args:
            x: input tensor of shape (N, C_in, T)
            mask: input mask of shape (N, 1, T)

        Returns:
            out: tuple containing L tensors of shape (N, C_out, T_l) where L is the number of stages.
        """
        z = self.encoder(x, mask)
        out: dict[str, tuple[Tensor, ...]] = {head_name: tuple() for head_name in self.heads.keys()}

        for z_l in z:
            z_l_heads = torch.split(z_l, self.head_splits, dim=1)
            for (head_name, head), z_l_head in zip(self.heads.items(), z_l_heads):
                out[head_name] += (head(z_l_head),)

        return out

        # out: dict[str, Tensor] = dict()
        # for z_l in z:
        #     out_l = dict()
        #     z_l_heads = torch.split(z_l, self.head_splits, dim=1)
        #     for (name, head), z_l_head in zip(self.heads.items(), z_l_heads):
        #         out_l[name] = head(z_l_head)
        #     out += (out_l,)
        # return out


def load_model(config: ModelConfig):
    model_id = config.name
    match model_id:
        case "mstcn":
            return RefinementLayersSegmentationModel(
                encoder_kwargs=config.encoder,
                heads_config=config.heads,
            )
        case _:
            raise ValueError(f"Unknown model: {model_id}")


if __name__ == "__main__":
    from pprint import pprint
    N, C, T = 3, 130, 256
    _x = torch.randn(N, C, T)
    _mask = torch.ones(N, 1, T, dtype=torch.bool)
    _model = load_model(
        ModelConfig(
            name="mstcn",
            encoder=dict(
                in_channels=130,
                out_channels=4,
                hidden_channels=64,
                n_stages=4,
                n_layers=10,
            ),
            heads=OrderedDict(
                classification=HeadConfig(
                    in_channels=2,
                    out_channels=2,
                    layer='identity',
                ),
                regression=HeadConfig(
                    in_channels=2,
                    out_channels=2,
                    layer='identity',
                ),
            ),
        )
    )
    _out = _model(_x, _mask)
    pprint({head_name: [out.shape for out in head_outs] for head_name, head_outs in _out.items()})
