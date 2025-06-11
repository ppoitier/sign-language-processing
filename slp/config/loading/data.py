from torch.utils.data import DataLoader

from slp.config.loading.codecs import load_segments_codec
from slp.config.templates.data import SegmentationDatasetConfig, RecognitionDatasetConfig
from slp.transforms.pose_pipelines import get_pose_pipeline
from slp.data.segmentation.densely_annotated import (
    DenselyAnnotatedSLDataset,
    Collator,
)
from slp.data.recognition.isolated_supervised import IsolatedSignsRecognition, ISLRCollator


def load_segmentation_dataset(config: SegmentationDatasetConfig) -> tuple[DenselyAnnotatedSLDataset, DataLoader | None]:
    use_windows, window_size, window_stride, max_empty_windows = False, 1500, 1200, None
    pose_transforms = None
    segment_codecs = {}
    if config.preprocessing is not None:
        use_windows = config.preprocessing.use_windows
        window_size = config.preprocessing.window_size
        window_stride = config.preprocessing.window_stride
        max_empty_windows = config.preprocessing.max_empty_windows
        pose_transforms = get_pose_pipeline(
            config.preprocessing.pose_transforms_pipeline
        )
        segment_codecs = {
            codec_config.name: load_segments_codec(codec_config)
            for codec_config in config.preprocessing.segment_codecs
        }
    dataset = DenselyAnnotatedSLDataset(
        url=config.shards_url,
        pose_transforms=pose_transforms,
        verbose=config.verbose,
        segment_codecs=segment_codecs,
        use_windows=use_windows,
        window_size=window_size,
        window_stride=window_stride,
        max_empty_windows=max_empty_windows,
    )
    if config.loader is None:
        return dataset, None
    loader_config = config.loader
    loader = DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        persistent_workers=True if loader_config.num_workers > 0 else False,
        collate_fn=Collator(segment_codec_names=list(segment_codecs.keys())),
        pin_memory=loader_config.pin_memory,
    )
    return dataset, loader


def load_recognition_dataset(config: RecognitionDatasetConfig) -> tuple[IsolatedSignsRecognition, DataLoader | None]:
    pose_transforms = None
    if config.preprocessing is not None:
        pose_transforms = get_pose_pipeline(
            config.preprocessing.pose_transforms_pipeline
        )
    dataset = IsolatedSignsRecognition(
        url=config.shards_url,
        pose_transforms=pose_transforms,
        verbose=config.verbose,
    )
    if config.loader is None:
        return dataset, None
    loader_config = config.loader
    loader = DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        persistent_workers=True if loader_config.num_workers > 0 else False,
        pin_memory=loader_config.pin_memory,
        collate_fn=ISLRCollator(min_length=64, flatten_poses=True),
    )
    return dataset, loader
