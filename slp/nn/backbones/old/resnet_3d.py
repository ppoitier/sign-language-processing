from typing import Optional

from pytorchvideo.models.hub import i3d_r50
import torch
from torch import nn, Tensor


class ResNet_3d(nn.Module):
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.i3d_r50 = i3d_r50(pretrained=False)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.i3d_r50.load_state_dict(checkpoint['model_state'])
        self.i3d_r50.blocks[-1].proj = nn.Identity()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x.transpose(1, 2).contiguous()
        """
        Args:
            x: tensor of shape (B, T, C_in, H, W)
            mask: boolean tensor shape (B, 1, T)

        Returns:
            ...
        """
        return self.i3d_r50(x)


if __name__ == '__main__':
    # checkpoint = torch.load("E:/weights/kinetic400/I3D_8x8_R50.pyth", map_location='cpu', weights_only=True)
    #
    # model = i3d_r50(pretrained=False)
    # model.load_state_dict(checkpoint['model_state'])
    from torchinfo import summary

    model = ResNet_3d("E:/weights/kinetic400/I3D_8x8_R50.pyth")
    N, C_out, T, H_out, W_out = 2, 3, 64, 224, 224
    x = torch.randn(N, T, C_out, H_out, W_out)
    mask = torch.ones(N, 1, T).bool()

    summary(model, input_data=(x, mask))

    # logits = model(x, mask)
    # print(logits.shape)
