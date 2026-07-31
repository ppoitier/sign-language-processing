from sign_language_tools.common.transforms import TransformTuple, Compose, Randomize
from sign_language_tools.pose.transforms import *


def stochastic_augments_with_flatten():
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
