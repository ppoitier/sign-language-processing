from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transform import Concatenate, Flatten, DropCoordinates

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
