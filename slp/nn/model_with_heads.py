from typing import Optional, Union

from torch import nn, Tensor


class Head(nn.Module):
    """
    A flexible head wrapper that applies a model to a list of feature tensors.

    It can slice a specific range of input channels for the model. The channel
    range can be the same for all feature tensors or specified individually for each.
    """

    def __init__(
        self,
        model: nn.Module,
        in_channels_range: Union[
            tuple[int, Optional[int]], list[tuple[int, Optional[int]]]
        ] = (0, None),
    ):
        super().__init__()
        self.model = model
        self.in_channels_range = in_channels_range

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Always accepts a list of tensors from different stages and returns a list of outputs.

        Args:
            features: A list of input tensors of shape (N, C_in, ...).

        Returns:
            A list of output tensors.
        """
        if isinstance(self.in_channels_range, list):
            if len(features) != len(self.in_channels_range):
                raise ValueError(
                    f"Number of feature tensors ({len(features)}) does not match "
                    f"number of channel ranges ({len(self.in_channels_range)})."
                )
            ranges = self.in_channels_range
        else:
            ranges = [self.in_channels_range] * len(features)

        outputs = []
        for z, (start, end) in zip(features, ranges):
            sliced_z = z[:, start:end]
            outputs.append(self.model(sliced_z))
        return outputs


class MultiHeadModel(nn.Module):
    """
    A unified model wrapper for single-stage or multi-stage backbones with multiple heads.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: nn.ModuleDict,
        neck: Optional[nn.Module] = None,
        predict_on_all_stages: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.neck = neck
        self.predict_on_all_stages = predict_on_all_stages

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> dict[str, Union[list[Tensor], Tensor]]:
        # 1. Get features from the backbone
        features = self.backbone(x, mask=mask)

        # 2. Normalize backbone output to always be a list of tensors
        if not isinstance(features, (list, tuple)):
            features = [features]

        # 3. Apply the neck, if it exists
        if self.neck is not None:
            features = self.neck(features)

        # 4. Select stages for prediction based on the flag
        if not self.predict_on_all_stages:
            features = features[-1:]

        # 5. Apply heads
        head_outputs = {}
        for head_name, head in self.heads.items():
            predictions = head(features)

            # 6. Unwrap the output from the list if it contains only one tensor.
            if len(predictions) == 1:
                head_outputs[head_name] = predictions[0]
            else:
                head_outputs[head_name] = predictions

        return head_outputs


# --- Dummy Modules for Demonstration ---
class DummyBackbone(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, multi_stage: bool = False):
        super().__init__()
        self.multi_stage = multi_stage
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels if idx == 0 else out_channels, out_channels, 3, padding=1)
            for idx in range(3)
        ])

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, list[Tensor]]:
        outputs = []
        for conv in self.convs:
            x = conv(x)
            outputs.append(x)

        if self.multi_stage:
            # Returns a list of tensors for multi-stage processing
            return outputs
        else:
            # Returns the final tensor for single-stage processing
            return outputs[-1]


class DummyHeadModel(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.layer = nn.Linear(in_channels, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # Global average pooling and linear layer
        return self.layer(x.mean(dim=-1))


if __name__ == "__main__":
    import torch

    N, C_in, T = 3, 128, 50
    C_hidden, C_out = 256, 2
    x = torch.randn(N, C_in, T)
    mask = torch.ones(N, 1, T).bool()
    backbone = DummyBackbone(C_in, C_hidden, multi_stage=True)
    dummy_head = DummyHeadModel(C_hidden//2, C_out)

    model = MultiHeadModel(
        backbone=backbone,
        heads=nn.ModuleDict(
            {
                "head1": Head(dummy_head, in_channels_range=(0, 128)),
                "head2": Head(dummy_head, in_channels_range=(128, 256)),
            }
        ),
    )
    logits = model(x, mask)
    print({k: [vv.shape for vv in v] for k, v in logits.items()})
