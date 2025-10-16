from torch.utils.data import DataLoader

from slp.data.densely_annotated import load_dataset, DenselyAnnotatedSLDataset
from slp.tasks.segmentation.config import SegmentationTaskConfig


def load_segmentation_datasets(
    config: SegmentationTaskConfig,
) -> tuple[dict[str, DenselyAnnotatedSLDataset], dict[str, DataLoader]]:
    datasets = {}
    dataloaders = {}
    for dataset_name, dataset_config in config.datasets.items():
        dataset, loader = load_dataset(dataset_config)
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = loader
    return datasets, dataloaders
