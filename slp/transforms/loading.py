from sign_language_tools.common.transforms import Compose

from slp.core.config.transform import TransformConfig
from slp.core.registry import POSE_TRANSFORM_REGISTRY, VIDEO_TRANSFORM_REGISTRY


def load_pose_transform(configs: list[TransformConfig]):
    if configs is None:
        return None
    transforms = []
    for config in configs:
        transform_cls = POSE_TRANSFORM_REGISTRY.get(config.name)
        transforms.append(transform_cls(**config.kwargs))
    return Compose(transforms)


def load_video_transform(configs: list[TransformConfig] | None):
    if configs is None:
        return None
    transforms = []
    for config in configs:
        transform_cls = VIDEO_TRANSFORM_REGISTRY.get(config.name)
        transforms.append(transform_cls(**config.kwargs))
    return Compose(transforms)
