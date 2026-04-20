from typing import Any

from sldl.targets.target import TargetEncoder
from sign_language_tools.annotations.transforms import (
    SegmentsToBoundaries,
    RemoveOverlapping,
)


class BoundaryTarget(TargetEncoder):
    def __init__(
        self, boundary_width: float | int = 4, annotation_id: str = "both_hands"
    ):
        super().__init__()
        self.boundary_width = boundary_width
        self.annotation_id = annotation_id
        # self.remove_overlapping = RemoveOverlapping()

    def encode(self, sample: dict) -> Any:
        segments = (
            sample["annotations"][self.annotation_id]
            .loc[:, ["start_frame", "end_frame"]]
            .values
        )
        segment_to_boundaries = SegmentsToBoundaries(
            width=self.boundary_width if isinstance(self.boundary_width, int) else None,
            relative_width=(
                self.boundary_width if isinstance(self.boundary_width, float) else None
            ),
            min_width=2,
            max_end=sample["n_frames"],
            boundary_label=(1, 2),
            # exclude_start=True,
            # exclude_end=True,
            # discrete_gap=True,
            # rounded=True,
        )
        segments = segment_to_boundaries(segments)
        return segments

    def collate(self, batch_targets: list[Any]) -> Any: ...
