from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transforms import (
    Concatenate,
    NormalizeByReferenceEdge,
    CenterOnLandmarks,
    ToRGBImage,
    TemporalCrop,
    Resample,
    DropCoordinates,
)

from slp.core.registry import POSE_TRANSFORM_REGISTRY


@POSE_TRANSFORM_REGISTRY.register("to-image")
class ToImage:
    def __init__(self):
        self.transform = Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                # NormalizeByReferenceEdge(ref_edge=(11, 12)),
                # CenterOnLandmarks((11, 12)),
                TemporalCrop(64),
                Resample(64, 'nearest'),
                ToRGBImage(),
            ]
        )

    def __call__(self, poses):
        return self.transform(poses)


@POSE_TRANSFORM_REGISTRY.register("to-image-drop-z")
class ToImageDropZ:
    def __init__(self):
        self.transform = Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                # NormalizeByReferenceEdge(ref_edge=(11, 12)),
                # CenterOnLandmarks((11, 12)),
                DropCoordinates('z'),
                TemporalCrop(64),
                Resample(64, 'nearest'),
                ToRGBImage(),
            ]
        )

    def __call__(self, poses):
        return self.transform(poses)
