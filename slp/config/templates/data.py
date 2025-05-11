from pydantic import BaseModel, Field


class DataLoaderConfig(BaseModel):
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True


class DataPreprocessing(BaseModel):
    pose_transforms_pipeline: str = 'none'


class ContinuousDataPreprocessing(DataPreprocessing):
    use_windows: bool = False
    window_size: int = 1500
    window_stride: int = 1200
    max_empty_windows: int | None = None


class SegmentCodecConfig(BaseModel):
    name: str
    use_offsets: bool = False
    args: dict = Field(default_factory=dict)


class SegmentationDataPreprocessing(ContinuousDataPreprocessing):
    segment_transforms_pipeline: str = 'none'
    segment_codecs: list[SegmentCodecConfig] = Field(default_factory=list)


class SegmentationDatasetConfig(BaseModel):
    shards_url: str
    mode: str = 'test'
    loader: DataLoaderConfig | None = None
    preprocessing: SegmentationDataPreprocessing | None = None
    verbose: bool = False