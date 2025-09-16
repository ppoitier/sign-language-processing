from torch import nn, Tensor
from torch.nn.functional import pad

from slp.config.templates.model import ModelConfig
from slp.nn.spoter import SPOTER

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SPOTER(
            n_classes=64,
            c_in=130,
            max_sequence_lengths=200,
            n_heads=10,
            n_encoder_layers=6,
            n_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
        )

    def forward(self, x: Tensor, masks: Tensor) -> dict[str, Tensor]:
        x = x.permute(0, 2, 1).contiguous()
        return {'classification': self.model(x)}


#
# class ClassificationModel(nn.Module):
#     def __init__(
#             self,
#             encoder: nn.Module,
#             embed_size: int,
#             max_length: int,
#             n_classes: int,
#     ):
#         super().__init__()
#         self.max_length = max_length
#         self.encoder = encoder
#         self.cls_head = nn.Linear(embed_size, n_classes)
#
#     def forward(self, x: Tensor, masks: Tensor):
#         """
#         Args:
#             x: tensor of shape (N, C_in, T)
#             masks: tensor of shape (N, 1, T)
#
#         Returns:
#             logits: tensor of shape (N, n_classes)
#         """
#         x = x[:, :, :self.max_length]
#         padding = self.max_length - x.shape[-1]
#         x = pad(x, (padding, 0), mode='constant', value=0.0)
#         masks = pad(masks, (padding, 0), mode='constant', value=0)
#         z = self.encoder(x.permute(0, 2, 1).contiguous(), masks.squeeze(1))
#         return {'classification': self.cls_head(z)}


def load_model(config: ModelConfig) -> nn.Module:
    match config.name:
        # case 'vit':
        #     encoder_args = config.encoder
        #     return ClassificationModel(
        #         encoder=ViT(**encoder_args),
        #         embed_size=encoder_args['out_channels'],
        #         max_length=encoder_args['max_length'],
        #         n_classes=config.heads['classification'].out_channels,
        #     )
        case 'spoter':
            return ClassificationModel()
        case _:
            raise ValueError(f"Unknown model: {config.name}")
