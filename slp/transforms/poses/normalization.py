from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transform import (
    Concatenate,
    Flatten,
    DropCoordinates,
    NormalizeEdgeLengths,
    CenterOnLandmarks,
)

from slp.core.registry import POSE_TRANSFORM_REGISTRY


@POSE_TRANSFORM_REGISTRY.register("norm+flatten2d")
class Normalized:
    def __init__(self):
        self.transform = Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                NormalizeEdgeLengths(ref_edge=(11, 12)),
                CenterOnLandmarks((11, 12)),
                DropCoordinates("z"),
                Flatten(),
            ]
        )

    def __call__(self, poses):
        return self.transform(poses)
