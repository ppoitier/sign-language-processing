import torch

from sldl import SignLanguageDataset

from slp.core.config.training import TrainingConfig
from slp.core.registry import CRITERION_REGISTRY
from slp.nn.losses.multi_layer import MultiLayerLoss
from slp.nn.losses.multi_task import MultiTaskLoss


def build_multi_layer_criterion(
    config: TrainingConfig,
    training_dataset: SignLanguageDataset,
) -> MultiTaskLoss:
    """
    Builds multitask, multi-layer loss criterion from the training configuration.
    """
    task_criteria = {}

    for head_name, criterion_config in config.loss_functions.items():
        criterion_cls = CRITERION_REGISTRY.get(criterion_config.name)
        weights = None
        if criterion_config.use_weights:
            print(
                f'Loading weights [{head_name}] with "{criterion_config.weight_strategy}" strategy...'
            )
            target_name = config.heads_to_targets[head_name]
            weights = training_dataset.get_label_weights(
                target_name, strategy=criterion_config.weight_strategy
            )
            weights = torch.tensor(
                [weights[i] for i in range(config.n_classes)],
                device="cuda",
                dtype=torch.float32,
            )
        criterion = criterion_cls(weights=weights, **criterion_config.kwargs)
        if config.is_output_multilayer:
            criterion = MultiLayerLoss(
                base_criterion=criterion,
                apply_on_all_stages=criterion_config.multi_layer,
            )
        task_criteria[head_name] = criterion

    return MultiTaskLoss(criteria=task_criteria)
