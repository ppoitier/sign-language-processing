from sldl.targets.frame_labels import FrameLabelsTarget
from sldl.targets.segments import SegmentTarget
from sldl.targets.temporal_boundary_offset import TemporalBoundaryOffsetsTarget

from slp.targets.bio_segmentation import BIOLabelsTarget
from slp.targets.boundaries import (
    SegmentBoundaryLabelsTarget,
    SegmentBoundaryOffsetsTarget,
    SegmentBoundarySegmentsTarget,
)

from sign_language_tools.annotations.transforms import RemoveOverlapping


def load_target(target_id: str):
    if target_id == "temporal-segmentation":
        return FrameLabelsTarget(segment_transform=RemoveOverlapping(min_gap=2))
    elif target_id == "temporal-offsets":
        return TemporalBoundaryOffsetsTarget()
    elif target_id == "segments":
        return SegmentTarget()
    elif target_id == "bio-tags":
        return BIOLabelsTarget()
    elif target_id == "boundaries-segments":
        return SegmentBoundarySegmentsTarget()
    elif target_id == "boundaries-segmentation":
        return SegmentBoundaryLabelsTarget(boundary_labels=(1, 1))
    elif target_id == "boundaries-offsets":
        return SegmentBoundaryOffsetsTarget()
    raise ValueError(f"Unknown target: {target_id}")