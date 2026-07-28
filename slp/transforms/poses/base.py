from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transforms import Concatenate, Flatten, DropCoordinates, TemporalCrop

from slp.core.registry import POSE_TRANSFORM_REGISTRY


@POSE_TRANSFORM_REGISTRY.register("flatten2d")
class FlattenPoseHands:
    def __init__(self):
        self.transform = Compose([
            Concatenate(['upper_pose', 'left_hand', 'right_hand']),
            DropCoordinates('z'),
            Flatten()
        ])

    def __call__(self, poses):
        return self.transform(poses)


@POSE_TRANSFORM_REGISTRY.register("islr+flatten3d")
class FlattenPoseHands3D:
    def __init__(self):
        self.transform = Compose([
            Concatenate(['upper_pose', 'left_hand', 'right_hand']),
            TemporalCrop(64),
            Flatten()
        ])

    def __call__(self, poses):
        return self.transform(poses)
