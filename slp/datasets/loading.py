from sldl import SignLanguageDataset
from sldl.targets.temporal_segmentation import TemporalSegmentationTarget
from sldl.targets.temporal_boundary_offset import TemporalBoundaryOffsetsTarget
from sldl.targets.segments import SegmentTarget
from torch.utils.data import DataLoader

from slp.core.config.dataset import ContinuousDatasetConfig, IsolatedDatasetConfig
from slp.datasets.dataloaders import load_dataloader
from slp.transforms.loading import load_pose_transform, load_video_transform


def load_target(target_id: str):
    if target_id == "temporal-segmentation":
        return TemporalSegmentationTarget()
    elif target_id == "temporal-offsets":
        return TemporalBoundaryOffsetsTarget()
    elif target_id == "segments":
        return SegmentTarget()
    raise ValueError(f"Unknown target: {target_id}")


def load_continuous_dataset(
    config: ContinuousDatasetConfig,
) -> SignLanguageDataset:
    targets = None
    use_windows, window_size, window_stride, max_empty_windows = False, 3500, 2800, None
    pose_transform, video_transform = None, None
    if config.preprocessing:
        targets = {
            target_id: load_target(target_id)
            for target_id in config.preprocessing.targets
        }
        use_windows = config.preprocessing.use_windows
        window_size = config.preprocessing.window_size
        window_stride = config.preprocessing.window_stride
        max_empty_windows = config.preprocessing.max_empty_windows
        pose_transform = load_pose_transform(config.preprocessing.pose_transforms)
        video_transform = load_video_transform(config.preprocessing.video_transforms)
    return SignLanguageDataset(
        shards_url=config.shards_url,
        targets=targets,
        annotations=('both_hands',),
        precompute_targets=True,
        use_windows=use_windows,
        window_size=window_size,
        window_stride=window_stride,
        max_empty_windows=max_empty_windows,
        pose_transform=pose_transform,
        video_transform=video_transform,
    )


def load_continuous_datasets_and_loaders(
    configs: dict[str, ContinuousDatasetConfig],
) -> tuple[dict[str, SignLanguageDataset], dict[str, DataLoader]]:
    assert 'training' in configs, "Missing 'training' dataset."
    assert 'validation' in configs, "Missing 'validation' dataset."
    assert 'testing' in configs, "Missing 'testing' dataset."
    datasets = {k: load_continuous_dataset(config) for k, config in configs.items()}
    dataloaders = {
        k: load_dataloader(datasets[k], config.dataloader)
        for k, config in configs.items()
        if config.dataloader is not None
    }
    return datasets, dataloaders


def load_isolated_dataset(
    config: IsolatedDatasetConfig,
) -> SignLanguageDataset:
    targets = None
    pose_transform, video_transform = None, None
    if config.preprocessing:
        targets = {
            target_id: load_target(target_id)
            for target_id in config.preprocessing.targets
        }
        pose_transform = load_pose_transform(config.preprocessing.pose_transforms)
        video_transform = load_video_transform(config.preprocessing.video_transforms)
    return SignLanguageDataset(
        shards_url=config.shards_url,
        isolated=True,
        targets=targets,
        annotations=None,
        precompute_targets=True,
        use_windows=False,
        pose_transform=pose_transform,
        video_transform=video_transform,
    )


def load_isolated_datasets_and_loaders(
    configs: dict[str, IsolatedDatasetConfig],
) -> tuple[dict[str, SignLanguageDataset], dict[str, DataLoader]]:
    assert 'training' in configs, "Missing 'training' dataset."
    assert 'validation' in configs, "Missing 'validation' dataset."
    assert 'testing' in configs, "Missing 'testing' dataset."
    datasets = {k: load_isolated_dataset(config) for k, config in configs.items()}
    dataloaders = {
        k: load_dataloader(datasets[k], config.dataloader)
        for k, config in configs.items()
        if config.dataloader is not None
    }
    return datasets, dataloaders
