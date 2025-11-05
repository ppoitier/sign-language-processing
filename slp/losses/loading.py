import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from slp.config.templates.training import TrainingConfig, CriterionConfig
from slp.losses.multi_layer_loss import MultiLayerLoss
from slp.losses.multihead import MultiHeadLoss
from slp.losses.smoothing import WithSmoothingLoss
from slp.losses.generalized_iou import GeneralizedIoU
from slp.losses.repeated_per_frame import RepeatedPerFrameLoss


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
        case "repeated":
            return RepeatedPerFrameLoss(
                load_criterion_by_id(sub_criterion, kwargs, weights)
            )
        case "gIoU":
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


def load_criterion(
    training_dataset,
    training_config: TrainingConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_weights = {}
    for name, config in training_config.criterion.items():
        if config.use_weights:
            weights = training_dataset.get_label_weights()
            print(f"Use weights for target:", weights)
            criterion_weights[name] = torch.from_numpy(weights).float().to(device)
    return load_multihead_criterion(training_config.criterion, criterion_weights)
