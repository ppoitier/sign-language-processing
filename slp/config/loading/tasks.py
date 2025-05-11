from slp.config.templates.task import SegmentationTaskConfig
from slp.config.loading.data import load_segmentation_dataset
from slp.config.loading.trainer import load_segmentation_lightning_module


def load_segmentation_task(config: SegmentationTaskConfig):
    task_id = config.id
    data_sets = {}
    data_loaders = {}
    for dataset_name, dataset_config in config.datasets.items():
        dataset, loader = load_segmentation_dataset(dataset_config)
        data_sets[dataset_name] = dataset
        data_loaders[dataset_name] = loader
    module = load_segmentation_lightning_module(config, training_dataset=data_sets.get('training'))
    return task_id, data_sets, data_loaders, module


