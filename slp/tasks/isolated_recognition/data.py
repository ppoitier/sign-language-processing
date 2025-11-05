from torch.utils.data import DataLoader

from slp.data.isolated_supervised import IsolatedSignsRecognitionDataset, load_dataset
from slp.tasks.isolated_recognition.config import IsolatedRecognitionTaskConfig


def load_isolated_recognition_datasets(
    config: IsolatedRecognitionTaskConfig,
) -> tuple[dict[str, IsolatedSignsRecognitionDataset], dict[str, DataLoader]]:
    datasets = {}
    dataloaders = {}
    for dataset_name, dataset_config in config.datasets.items():
        dataset, loader = load_dataset(dataset_config)
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = loader
    return datasets, dataloaders
