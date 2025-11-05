import torch
from torchvision.transforms import v2 as T

from sign_language_tools.video.transforms import TemporalCrop, TemporalPad


def get_video_transform_pipeline(pipeline_name: str):
    match pipeline_name:
        case "wlasl-training":
            return T.Compose([
                TemporalCrop(max_width=64, location='center'),
                TemporalPad(min_width=64, location='end'),
                T.ToDtype(torch.float32, scale=True),  # Now [0.0, 1.0]
                T.Resize(256, antialias=True),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Now [-1.0, 1.0]
            ])
        case 'wlasl-testing':
            return T.Compose([
                TemporalCrop(max_width=64, location='center'),
                TemporalPad(min_width=64, location='end'),
                T.ToDtype(torch.float32, scale=True),  # Now [0.0, 1.0]
                T.Resize(256, antialias=True),
                T.CenterCrop(224),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Now [-1.0, 1.0]
            ])
        case "none":
            return T.Identity()
        case _:
            raise ValueError(f"Unknown pipeline: {pipeline_name}.")
