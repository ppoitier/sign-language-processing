from torch import Tensor
from torch.nn import CrossEntropyLoss

from slp.config.templates.training import CriterionConfig
from slp.losses.multi_layer_loss import MultiLayerLoss
from slp.losses.multihead import MultiHeadLoss
from slp.losses.smoothing import WithSmoothingLoss
from slp.losses.generalized_iou import GeneralizedIoU


def load_criterion_by_id(criterion_id: str, kwargs, weights: None | Tensor = None):
    criterion_components = criterion_id.split("+", maxsplit=1)
    curr_criterion = criterion_components[0]
    sub_criterion = criterion_components[1] if len(criterion_components) > 1 else None
    match curr_criterion:
        case "multi-layer":
            return MultiLayerLoss(load_criterion_by_id(sub_criterion, kwargs, weights))
        case "smoothing":
            return WithSmoothingLoss(
                load_criterion_by_id(sub_criterion, kwargs, weights)
            )
        case "cross-entropy":
            return CrossEntropyLoss(weight=weights, ignore_index=-1)
        case 'gIoU':
            return GeneralizedIoU()
        case _:
            raise ValueError(f"Unknown criterion: {curr_criterion}")


def load_multihead_criterion(
    configs: dict[str, CriterionConfig], label_weights: dict[str, Tensor]
) -> MultiHeadLoss:
    loss_functions = {
        name: load_criterion_by_id(config.name, config.kwargs, label_weights.get(name))
        for name, config in configs.items()
    }
    return MultiHeadLoss(loss_functions)
