from torch import nn, Tensor


class HydraModel(nn.Module):
    """
    A unified model wrapper for backbones with multiple prediction heads.

    The pipeline executes sequentially: Backbone -> Neck -> Heads.
    Outputs from the neck are passed to every head in the module dictionary.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: nn.ModuleDict,
        neck: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.neck = nn.Identity() if neck is None else neck

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> dict[str, list[Tensor]]:
        """
        Args:
            x: Input tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            A dictionary mapping head names to their respective lists of output tensors.
        """
        # 1. Get features from the backbone
        features = self.backbone(x, mask)

        # 2. Normalize backbone output to always be a list of tensors
        if not isinstance(features, (list, tuple)):
            features = [features]

        # 3. Apply the neck
        features = self.neck(features)

        # 4. Apply all heads
        return {
            head_name: head(features)
            for head_name, head in self.heads.items()
        }