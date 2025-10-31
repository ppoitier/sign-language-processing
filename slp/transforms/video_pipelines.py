import torch
from torchvision.transforms import v2

from sign_language_tools.video.transforms import TemporalRandomCrop, TemporalPad


def get_wlasl_video_transforms():
    n_frames = 64
    resize_dim = 256
    crop_dim = 224

    return v2.Compose([
        TemporalRandomCrop(n_frames),
        TemporalPad(n_frames),
        v2.Resize(size=(resize_dim, resize_dim), antialias=True),

        # 4. Add other spatial augmentations (optional but recommended)
        # v2.RandomCrop(size=(crop_dim, crop_dim)),
        # v2.RandomHorizontalFlip(p=0.5),

        # 5. Normalize to [-1, 1]
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
