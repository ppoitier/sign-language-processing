from pydantic import BaseModel, Field

from slp.config.templates.codec import SegmentCodecConfig


class DataLoaderConfig(BaseModel):
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    flatten_pose: bool = True


class DataPreprocessing(BaseModel):
    pose_transforms_pipeline: str = 'none'
    input_type: str = 'poses'


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
    vocab_size: int | None = None
    preprocessing: DataPreprocessing | None = None