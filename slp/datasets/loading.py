from sldl import SignLanguageDataset
from sldl.targets.temporal_segmentation import TemporalSegmentationTarget
from sldl.targets.temporal_boundary_offset import TemporalBoundaryOffsetsTarget
from torch.utils.data import DataLoader

from slp.core.config.dataset import ContinuousDatasetConfig
from slp.datasets.dataloaders import load_dataloader


def load_target(target_id: str):
    if target_id == "temporal-segmentation":
        return TemporalSegmentationTarget()
    elif target_id == "temporal-offsets":
        return TemporalBoundaryOffsetsTarget()
    raise ValueError(f"Unknown target: {target_id}")


def load_continuous_dataset(
    config: ContinuousDatasetConfig,
) -> SignLanguageDataset:
    targets = None
    use_windows, window_size, window_stride, max_empty_windows = False, 3500, 2800, None
    if config.preprocessing:
        targets = {
            target_id: load_target(target_id)
            for target_id in config.preprocessing.targets
        }
        use_windows = config.preprocessing.use_windows
        window_size = config.preprocessing.window_size
        window_stride = config.preprocessing.window_stride
        max_empty_windows = config.preprocessing.max_empty_windows
    return SignLanguageDataset(
        shards_url=config.shards_url,
        targets=targets,
        precompute_targets=True,
        use_windows=use_windows,
        window_size=window_size,
        window_stride=window_stride,
        max_empty_windows=max_empty_windows,
    )


def load_continuous_datasets_and_loaders(
    configs: dict[str, ContinuousDatasetConfig],
) -> tuple[dict[str, SignLanguageDataset], dict[str, DataLoader]]:
    datasets = {k: load_continuous_dataset(config) for k, config in configs.items()}
    #  TODO: enforce keys here (training, ...)
    dataloaders = {
        k: load_dataloader(datasets[k], config.dataloader)
        for k, config in configs.items()
        if config.dataloader is not None
    }
    return datasets, dataloaders
