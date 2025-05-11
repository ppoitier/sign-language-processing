from slp.config.templates.training import CriterionConfig
from slp.nn.losses.smoothing import MultiLayerClassificationLossWithSmoothing, ClassificationLossWithSmoothing
from slp.nn.losses.offsets import MultiLayerClassificationWithOffsetsLoss
from slp.nn.losses.cross_entropy import CrossEntropyLoss
from slp.nn.losses.generalized_iou import GeneralizedIoU


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
