from torch import nn, Tensor


class MultiLayerClassificationWithOffsetsLoss(nn.Module):
    def __init__(
        self,
        cls_loss_fn: nn.Module,
        reg_loss_fn: nn.Module,
        n_classes: int,
        reg_loss_coef: float = 1.0,
        return_loss_components: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.reg_loss_coef = reg_loss_coef
        self.return_loss_components = return_loss_components

        self.cls_loss_fn = cls_loss_fn
        self.reg_loss_fn = reg_loss_fn

    def forward(
        self, multilayer_logits: Tensor, targets: Tensor
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        cls_loss = 0
        reg_loss = 0

        cls_targets = targets[:, :, 0].long()
        reg_targets = targets[:, :, 1:]

        for logits in multilayer_logits.unbind(0):
            cls_logits = logits[:, :, : self.n_classes]
            cls_loss += self.cls_loss_fn(cls_logits, cls_targets)
            reg_logits = logits[:, :, self.n_classes :]
            reg_loss += self.reg_loss_fn(reg_logits, reg_targets)

        loss = cls_loss + self.reg_loss_coef * reg_loss
        if self.return_loss_components:
            return loss, cls_loss, reg_loss
        return loss
