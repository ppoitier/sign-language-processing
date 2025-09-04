from torch import nn, Tensor


class MultiHeadLoss(nn.Module):
    def __init__(self, loss_functions: dict[str, nn.Module]):
        super().__init__()
        self.loss_functions = loss_functions

    def forward(self, multihead_logits: dict[str, Tensor], multihead_targets: dict[str, Tensor]) -> Tensor:
        loss = 0.0
        for target_name in multihead_targets:
            loss += self.loss_functions[target_name](multihead_logits[target_name], multihead_targets[target_name])
        return loss
