from sign_language_tools.common.transforms import TransformTuple, Compose, Randomize
from sign_language_tools.pose.transforms import *

from slp.core.registry import POSE_TRANSFORM_REGISTRY


def stochastic_augments():
    return TransformTuple(
        Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                DropCoordinates("z"),
                TemporalRandomCrop(size=64),
                Randomize(GaussianNoise(0.002), probability=0.6),
                Randomize(HorizontalFlip(), probability=0.3),
                Randomize(RandomRotation2D(angle_range=(-0.3, 0.3)), probability=0.6),
                Randomize(
                    RandomTranslation(dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)),
                    probability=0.6,
                ),
                Randomize(RandomScale(min_scale=0.5, max_scale=1.5), probability=0.2),
                # Clip(),
                # NormalizeByReferenceEdge(ref_edge=(11, 12)),
                # CenterOnLandmarks((11, 12)),
                Split(
                    {
                        "upper_pose": (0, 23),
                        "left_hand": (23, 44),
                        "right_hand": (44, 65),
                    }
                ),
            ]
        )
    )


@POSE_TRANSFORM_REGISTRY.register("simclr-ready")
def stochastic_augments_model_ready():
    return TransformTuple(
        Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                NormalizeByReferenceEdge(ref_edge=(11, 12)),
                CenterOnLandmarks((11, 12)),
                DropCoordinates("z"),
                TemporalRandomCrop(size=64),
                Randomize(GaussianNoise(0.002), probability=0.6),
                Randomize(HorizontalFlip(), probability=0.3),
                Randomize(RandomRotation2D(angle_range=(-0.3, 0.3)), probability=0.6),
                Randomize(
                    RandomTranslation(dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)),
                    probability=0.6,
                ),
                Randomize(RandomScale(min_scale=0.5, max_scale=1.5), probability=0.2),
                Clip(),
                Flatten(),
            ]
        )
    )
