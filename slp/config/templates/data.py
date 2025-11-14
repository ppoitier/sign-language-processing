import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from slp.config.templates.codec import SegmentCodecConfig


class DataLoaderConfig(BaseModel):
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    flatten_pose: bool = True


class DataPreprocessing(BaseModel):
    pose_transform_pipeline: str = 'none'
    video_transform_pipeline: str = 'none'
    input_type: str = 'poses'
    include_videos: bool = False


class ContinuousDataPreprocessing(DataPreprocessing):
    use_windows: bool = False
    window_size: int = 1500
    window_stride: int = 1200
    max_empty_windows: int | None = None


class SegmentationDataPreprocessing(ContinuousDataPreprocessing):
    segment_transforms_pipeline: str = 'none'
    segment_codecs: list[SegmentCodecConfig] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    shards_url: str
    mode: str = 'test'
    loader: DataLoaderConfig | None = None
    verbose: bool = False


class SegmentationDatasetConfig(DatasetConfig):
    preprocessing: SegmentationDataPreprocessing | None = None


class RecognitionDatasetConfig(DatasetConfig):
    preprocessing: DataPreprocessing | None = None
    split_filepath: Optional[str] = None
    label_mapping_filepath: Optional[str] = None
    video_tar_path: Optional[str] = None
    video_tar_index_path: Optional[str] = None
    video_gpu_decoding: bool = False


    @field_validator('split_filepath', 'label_mapping_filepath', 'video_tar_path', 'video_tar_index_path', mode='after')
    def validate_filepaths(cls, filepath: Optional[str]):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File [{filepath}] not found.')
        return filepath
