from torch.utils.data import DataLoader

from slp.data.isolated_supervised import IsolatedSignsRecognitionDataset, load_dataset


def load_isolated_recognition_datasets(dataset_configs: dict) -> tuple[dict[str, IsolatedSignsRecognitionDataset], dict[str, DataLoader]]:
    datasets = {}
    dataloaders = {}
    for dataset_name, dataset_config in dataset_configs.items():
        dataset, loader = load_dataset(dataset_config)
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = loader
    return datasets, dataloaders
