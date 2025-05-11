from torch import nn, Tensor


class MultiLayerLoss(nn.Module):
    def __init__(self, single_layer_loss: nn.Module):
        super().__init__()
        self.loss = single_layer_loss

    def forward(self, multilayer_logits: Tensor, targets: Tensor):
        loss = 0
        for logits in multilayer_logits.unbind(0):
            loss += self.loss(logits, targets)
        return loss
