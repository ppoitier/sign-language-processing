import torch
from torch import nn, Tensor


class TaskChannelSplitter(nn.Module):
    """
    Splits the channel dimension of a feature tensor and routes the
    resulting chunks to task-specific heads.

    Args:
        split_sections: A list of integers defining the number of channels for each task.
        heads: A dictionary mapping task names to their respective head modules.
    """

    def __init__(self, split_sections: list[int], heads: dict[str, nn.Module]):
        super().__init__()
        self.split_sections = split_sections
        self.heads = nn.ModuleDict(heads)

        if len(self.split_sections) != len(self.heads):
            raise ValueError("Number of split sections must match the number of heads.")

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: A feature tensor of shape (N, C_out, T_l).

        Returns:
            A dictionary mapping task names to their output tensors.
        """
        # Split along the channel dimension
        splits = torch.split(x, self.split_sections, dim=1)

        outputs = {}
        for i, (task_name, head_module) in enumerate(self.heads.items()):
            outputs[task_name] = head_module(splits[i])

        return outputs
