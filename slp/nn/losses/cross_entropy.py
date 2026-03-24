from torch import nn

from slp.core.registry import CRITERION_REGISTRY


@CRITERION_REGISTRY.register("cross-entropy")
class CrossEntropyCriterion(nn.Module):
    def __init__(self, weights=None, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, **kwargs)

    def forward(self, input, target):
        return self.ce(input, target)

