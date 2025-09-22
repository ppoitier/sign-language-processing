from torchvision.transforms import ToTensor

from sign_language_tools.common.transforms import Compose, Randomize, Identity
from sign_language_tools.pose.transform import (
    Concatenate,
    DropCoordinates,
    NormalizeEdgeLengths,
    CenterOnLandmarks,
    Flatten,
    TemporalRandomCrop,
    GaussianNoise,
    HorizontalFlip,
    RandomRotation2D,
    RandomTranslation,
    RandomScale,
    Split,
    RandomTemporalScale,
    ToRGBImage,
    Resample,
    TemporalCrop,
    FixedResolutionNormalization,
    Padding,
)


def normalization_transforms():
    return Compose(
        [
            DropCoordinates("z"),
            NormalizeEdgeLengths(ref_edge=(11, 12)),
            CenterOnLandmarks((11, 12)),
        ]
    )


def get_pose_pipeline(pipeline_name: str):
    if pipeline_name == "none":
        return Identity()
    elif pipeline_name == "norm":
        return normalization_transforms()
    elif pipeline_name == "concat":
        return Concatenate(['upper_pose', 'left_hand', 'right_hand'])
    elif pipeline_name == "resample":
        return Resample(new_length=64, method='nearest')
    elif pipeline_name == "temporal-crop":
        return TemporalCrop(size=200, location='start')
    elif pipeline_name == "img":
        return Compose([
            ToRGBImage(),
            ToTensor(),
        ])
    elif pipeline_name == "split":
        return Split(
            {
                "upper_pose": (0, 23),
                "left_hand": (23, 44),
                "right_hand": (44, 65),
            }
        )
    elif pipeline_name == 'dropz':
        return DropCoordinates('z')
    elif pipeline_name == "flatten":
        return Flatten()
    elif pipeline_name == "nfrts":
        return random_nfrts()
    elif pipeline_name == "dilation":
        return random_time_dilation()
    elif pipeline_name == "wlasl-pose":
        n_samples = 50
        return Compose([
            Concatenate(['upper_pose', 'left_hand', 'right_hand']),
            DropCoordinates('z'),
            Padding(min_length=n_samples, mode='edge'),
            TemporalCrop(size=n_samples, location='center'),
            FixedResolutionNormalization(width=256, height=256),
            Flatten('features'),
        ])
    components = pipeline_name.split("+")
    if len(components) < 2:
        raise ValueError(f"Invalid pipeline: {pipeline_name}")
    return Compose([get_pose_pipeline(name) for name in components])


def random_nfrts():
    return Compose(
        [
            TemporalRandomCrop(size=64),
            Randomize(GaussianNoise(0.005), probability=0.6),
            Randomize(HorizontalFlip(origin=(0, 0)), probability=0.3),
            Randomize(RandomRotation2D(angle_range=(-0.3, 0.3)), probability=0.6),
            Randomize(
                RandomTranslation(dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)),
                probability=0.6,
            ),
            Randomize(RandomScale(min_scale=0.5, max_scale=1.5), probability=0.2),
        ]
    )


def random_time_dilation():
    return Randomize(RandomTemporalScale(min_scale=0.5, max_scale=1.5), probability=0.6)
