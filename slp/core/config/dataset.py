from pydantic import BaseModel, Field


from slp.core.config.transform import TransformConfig


class DataLoaderConfig(BaseModel):
    batch_size: int = 16
    shuffle: bool = True
    n_workers: int = 0
    pin_memory: bool = True


class DataPreprocessing(BaseModel):
    targets: list[str] = Field(default_factory=list)
    pose_transforms: list[TransformConfig] | None = None
    video_transforms: list[TransformConfig] | None = None


class DatasetConfig(BaseModel):
    shards_url: str
    dataloader: DataLoaderConfig | None = None
    include_videos: bool = False
    preprocessing: DataPreprocessing | None = None


class ContinuousDataPreprocessing(DataPreprocessing):
    annotation_transform_pipeline: str = "none"
    use_windows: bool = False
    window_size: int = 1500
    window_stride: int = 1200
    max_empty_windows: int | None = None


class ContinuousDatasetConfig(DatasetConfig):
    preprocessing: ContinuousDataPreprocessing | None = None


class IsolatedDatasetConfig(DatasetConfig):
    ...


# class IsolatedRecognitionDatasetConfig(DatasetConfig):
#     preprocessing: DataPreprocessing | None = None
#     split_filepath: Optional[str] = None
#     label_mapping_filepath: Optional[str] = None
#     video_tar_path: Optional[str] = None
#     video_tar_index_path: Optional[str] = None
#     video_gpu_decoding: bool = False
#
#
#     @field_validator('split_filepath', 'label_mapping_filepath', 'video_tar_path', 'video_tar_index_path', mode='after')
#     def validate_filepaths(cls, filepath: Optional[str]):
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f'File [{filepath}] not found.')
#         return filepath
