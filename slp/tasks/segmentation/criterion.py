import torch

from slp.data.densely_annotated import DenselyAnnotatedSLDataset
from slp.config.templates.training import TrainingConfig
from slp.losses.loading import load_multihead_criterion


def load_segmentation_criterion(
        training_dataset: DenselyAnnotatedSLDataset,
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
