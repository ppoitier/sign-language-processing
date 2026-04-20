from torch import nn, Tensor


class SharedChannelHeads(nn.Module):
    """
    Routes the full feature tensor to every task-specific head.
    All heads receive the same channels (no splitting).

    Args:
        task_specs: A dictionary mapping task names to head modules.
    """

    def __init__(self, task_specs: dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(task_specs)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: A feature tensor of shape (N, C_out, T_l).

        Returns:
            A dictionary mapping task names to their output tensors.
        """
        return {name: head(x) for name, head in self.heads.items()}