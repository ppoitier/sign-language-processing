from torch import nn, Tensor, exp
from torch.nn.functional import cross_entropy


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,  # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        weight: Tensor = None,  # Manual rescaling weight given to each class
        reduction: str = "mean",  # PyTorch reduction to apply to the output
        **kwargs,
    ):
        """ Applies Focal Loss: https://arxiv.org/pdf/1708.02002.pdf """
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(self, inp: Tensor, targ: Tensor) -> Tensor:
        """ Applies focal loss based on https://arxiv.org/pdf/1708.02002.pdf """
        ce_loss = cross_entropy(inp, targ, weight=self.weight, reduction="none", **self.kwargs)
        p_t = exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss