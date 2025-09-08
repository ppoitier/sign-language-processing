from torch import nn, Tensor
import torch.nn.functional as F


class SmoothingLoss(nn.Module):
    def __init__(self, theta: float = 16):
        super().__init__()
        self.theta = theta

    def forward(self, logits: Tensor) -> Tensor:
        """
        Args:
            logits: tensor of shape (N, C_out, T)

        Returns:
            loss
        """
        # TODO: add masking
        return (
            F.mse_loss(
                logits[:, :, 1:].log_softmax(dim=1),
                logits.detach()[:, :, :-1].log_softmax(dim=1),
                reduction='none',
            )
            .clamp(min=0, max=self.theta)
            .mean()
        )


class WithSmoothingLoss(nn.Module):
    def __init__(
            self,
            loss_fn: nn.Module,
            smoothing_coef: float = 0.15,
            smoothing_theta: float = 16,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.smoothing_coef = smoothing_coef
        self.smoothing_loss = SmoothingLoss(theta=smoothing_theta)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.loss_fn(logits, targets) + self.smoothing_coef * self.smoothing_loss(logits)


# class ClassificationLossWithSmoothing(nn.Module):
#     def __init__(
#         self,
#         cls_loss_fn: nn.Module,
#         return_loss_components: bool = False,
#         smoothing_coef: float = 0.15,
#         smoothing_theta: float = 16,
#     ):
#         super().__init__()
#         self.return_loss_components = return_loss_components
#         self.smoothing_coef = smoothing_coef
#
#         self.cls_loss_fn = cls_loss_fn
#         self.smoothing_loss_fn = SmoothingLoss(smoothing_theta)
#
#     def forward(self, logits: Tensor, targets: Tensor):
#         cls_loss = self.cls_loss_fn(logits, targets)
#         smoothing_loss = self.smoothing_loss_fn(logits)
#         loss = cls_loss + self.smoothing_coef * smoothing_loss
#         if self.return_loss_components:
#             return loss, cls_loss, smoothing_loss
#         return loss
#
#
# class MultiLayerClassificationLossWithSmoothing(nn.Module):
#     def __init__(
#             self,
#             cls_loss_fn: nn.Module,
#             return_loss_components: bool = False,
#             smoothing_coef: float = 0.15,
#             smoothing_theta: float = 16,
#     ):
#         super().__init__()
#         self.single_layer_loss_fn = ClassificationLossWithSmoothing(
#             cls_loss_fn=cls_loss_fn,
#             return_loss_components=return_loss_components,
#             smoothing_coef=smoothing_coef,
#             smoothing_theta=smoothing_theta,
#         )
#         self.return_loss_components = return_loss_components
#
#     def forward(self, multilayer_logits: Tensor, targets: Tensor):
#         if self.return_loss_components:
#             loss = (0, 0, 0)
#         else:
#             loss = 0
#         for logits in multilayer_logits.unbind(0):
#             loss += self.single_layer_loss_fn(logits, targets)
#         return loss