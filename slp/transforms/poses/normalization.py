from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transforms import (
    Concatenate,
    Flatten,
    DropCoordinates,
    NormalizeByReferenceEdge,
    CenterOnLandmarks,
)

from slp.core.registry import POSE_TRANSFORM_REGISTRY


@POSE_TRANSFORM_REGISTRY.register("norm+flatten2d")
class Normalized:
    def __init__(self, body_parts: tuple[str, ...] = ("upper_pose", "left_hand", "right_hand")):
        self.transform = Compose(
            [
                Concatenate(body_parts),
                NormalizeByReferenceEdge(ref_edge=(11, 12)),
                CenterOnLandmarks((11, 12)),
                DropCoordinates("z"),
                Flatten(),
            ]
        )

    def __call__(self, poses):
        return self.transform(poses)
