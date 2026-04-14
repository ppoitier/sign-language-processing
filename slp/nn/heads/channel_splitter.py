import torch
from torch import nn, Tensor


class TaskChannelSplitter(nn.Module):
    """
    Splits the channel dimension of a feature tensor and routes the
    resulting chunks to task-specific heads.

    Args:
        task_specs: A dictionary mapping task names to a (n_channels, head_module) tuple.
            The iteration order of this dict defines the channel slicing order.
    """

    def __init__(self, task_specs: dict[str, tuple[int, nn.Module]]):
        super().__init__()
        self.task_names = list(task_specs.keys())
        self.split_sections = [n for n, _ in task_specs.values()]
        self.heads = nn.ModuleDict({name: head for name, (_, head) in task_specs.items()})

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: A feature tensor of shape (N, C_out, T_l).

        Returns:
            A dictionary mapping task names to their output tensors.
        """
        splits = torch.split(x, self.split_sections, dim=1)
        return {name: self.heads[name](chunk) for name, chunk in zip(self.task_names, splits)}
