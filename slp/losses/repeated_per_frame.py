from torch import nn, Tensor


class RepeatedPerFrameLoss(nn.Module):
    def __init__(self, sub_criterion: nn.Module):
        super(RepeatedPerFrameLoss, self).__init__()
        self.sub_criterion = sub_criterion

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: tensor of shape (N, C_out, T, ...)
            targets: tensor of shape (N)

        Returns:
            loss: scalar tensor
        """
        T = logits.size(2)
        targets = targets.unsqueeze(1).expand(-1, T)
        return self.sub_criterion(logits, targets)
