from torch import nn
from pytorch_metric_learning.losses import SupConLoss

from slp.config.templates.training import CriterionConfig
from slp.losses.smoothing import MultiLayerClassificationLossWithSmoothing, ClassificationLossWithSmoothing
from slp.losses.offsets import MultiLayerClassificationWithOffsetsLoss
from slp.losses import CrossEntropyLoss
from slp.losses.generalized_iou import GeneralizedIoU


def load_segmentation_criterion(config: CriterionConfig, criterion_weights=None):
    name = config.name
    if name == 'multi-layer+ce+smoothing':
        return MultiLayerClassificationLossWithSmoothing(
            cls_loss_fn=CrossEntropyLoss(weights=criterion_weights),
            return_loss_components=False,
        )
    elif name == "multi-layer+ce+smoothing+offsets":
        return MultiLayerClassificationWithOffsetsLoss(
            cls_loss_fn=ClassificationLossWithSmoothing(
                CrossEntropyLoss(weights=criterion_weights)
            ),
            reg_loss_fn=GeneralizedIoU(),
            n_classes=config.n_classes,
            return_loss_components=True,
        )
    else:
        raise ValueError(f'Unknown criterion: {name}.')


def load_recognition_criterion(config: CriterionConfig, criterion_weights=None):
    name = config.name
    if name == 'ce':
        return nn.CrossEntropyLoss(weight=criterion_weights)
    else:
        raise ValueError(f'Unknown criterion: {name}.')


def load_contrastive_criterion(config: CriterionConfig, criterion_weights=None):
    name = config.name
    if name == 'supcon':
        return SupConLoss(**config.kwargs)
    else:
        raise ValueError(f'Unknown criterion: {name}.')
