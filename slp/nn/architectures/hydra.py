from torch import nn, Tensor


class HydraModel(nn.Module):
    """
    A unified model wrapper for backbones with multiple prediction heads.

    The pipeline executes sequentially: Backbone -> Neck -> Head Assembly.
    The Head Assembly handles channel splitting and multi-stage distribution.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_assembly: nn.Module,
        neck: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head_assembly = head_assembly
        self.neck = nn.Identity() if neck is None else neck

    def forward(self, x: Tensor, mask: Tensor | None = None) -> dict[str, list[Tensor]]:
        """
        Args:
            x: Input tensor of shape (N, C_in, T).
            mask: Valid length mask of shape (N, 1, T).

        Returns:
            A dictionary mapping head names to their respective lists of multi-stage output tensors.
        """
        # 1. Get features from the backbone
        features = self.backbone(x, mask)

        # 2. Normalize backbone output to always be a list of tensors
        if not isinstance(features, (list, tuple)):
            features = [features]

        # 3. Apply the neck
        features = self.neck(features)

        # 4. Apply the integrated head assembly (Splitter + MultiStage Wrapper)
        # Returns a list of dictionaries: [{'slr': t1, 'sls': t1}, {'slr': t2, 'sls': t2}]
        step_predictions = self.head_assembly(features)

        # 5. Transpose to a dictionary of lists: {'slr': [t1, t2], 'sls': [t1, t2]}
        transposed = {
            task_name: [step[task_name] for step in step_predictions]
            for task_name in step_predictions[0].keys()
        }

        return transposed
