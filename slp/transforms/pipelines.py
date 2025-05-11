from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transform import (
    Concatenate,
    DropCoordinates,
    NormalizeEdgeLengths,
    CenterOnLandmarks,
    Flatten,
)


def normalize_and_flatten_pipeline():
    return Compose(
        [
            Concatenate(["upper_pose", "left_hand", "right_hand"]),
            DropCoordinates("z"),
            NormalizeEdgeLengths(unitary_edge=(11, 12)),
            CenterOnLandmarks((11, 12)),
            Flatten(),
        ]
    )
