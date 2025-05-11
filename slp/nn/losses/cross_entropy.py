from torch import nn, Tensor
from torch.nn.functional import cross_entropy


class CrossEntropyLoss(nn.Module):
    def __init__(self, weights: Tensor | None = None):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        return cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            targets.long(),
            ignore_index=-1,
            weight=self.weights,
        )
