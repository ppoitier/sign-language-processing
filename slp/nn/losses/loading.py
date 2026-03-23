from torch import Tensor

from slp.core.config.training import TrainingConfig, CriterionConfig

from slp.core.registry import CRITERION_REGISTRY
from slp.nn.losses.multi_layer import MultiLayerLoss
from slp.nn.losses.multi_task import MultiTaskLoss


def build_multi_layer_loss(
    config: TrainingConfig, weights: dict[str, Tensor] | None = None
) -> MultiTaskLoss:
    """
    Builds multitask, multi-layer loss criterion from the training configuration.
    """
    task_criteria = {}

    for task_name, criterion_config in config.loss_functions.items():
        criterion_cls = CRITERION_REGISTRY.get(criterion_config.name)
        criterion = criterion_cls(weights=weights, **criterion_config.kwargs)
        if config.is_output_multilayer:
            criterion = MultiLayerLoss(
                base_criterion=criterion,
                apply_on_all_stages=criterion_config.multi_layer,
            )
        task_criteria[task_name] = criterion

    return MultiTaskLoss(criteria=task_criteria)
