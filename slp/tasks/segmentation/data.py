from torch.utils.data import DataLoader

from slp.data.densely_annotated import load_dataset, DenselyAnnotatedSLDataset
from slp.config.templates.data import SegmentationDatasetConfig


def load_segmentation_datasets(configs: dict[str, SegmentationDatasetConfig]) -> tuple[dict[str, DenselyAnnotatedSLDataset], dict[str, DataLoader]]:
    datasets = {}
    dataloaders = {}
    for dataset_name, dataset_config in configs.items():
        dataset, loader = load_dataset(dataset_config)
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = loader
    return datasets, dataloaders
