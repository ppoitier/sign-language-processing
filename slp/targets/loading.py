from sldl.targets.frame_labels import FrameLabelsTarget
from sldl.targets.segments import SegmentTarget
from sldl.targets.temporal_boundary_offset import TemporalBoundaryOffsetsTarget

from slp.targets.bio_segmentation import BIOLabelsTarget
from slp.targets.boundaries import (
    SegmentBoundaryLabelsTarget,
    SegmentBoundaryOffsetsTarget,
    SegmentBoundarySegmentsTarget,
)

from sign_language_tools.common.transforms import Compose, Identity
from sign_language_tools.annotations.transforms import (
    RemoveOverlapping,
    CloseShortSilences,
    RandomRelativeScaleSegments,
    RandomRelativeMoveSegments,
)


def load_continuous_target(target_id: str, variant=None):
    if variant == "low-noise":
        annot_transform = Compose(
            [
                RandomRelativeMoveSegments(dx_std=0.05),
                RandomRelativeScaleSegments(scale_std=0.1),
            ]
        )
    elif variant == "medium-noise":
        annot_transform = Compose(
            [
                RandomRelativeMoveSegments(dx_std=0.1),
                RandomRelativeScaleSegments(scale_std=0.3),
            ]
        )
    elif variant == "high-noise":
        annot_transform = Compose(
            [
                RandomRelativeMoveSegments(dx_std=0.1),
                RandomRelativeScaleSegments(scale_std=0.5),
            ]
        )
    else:
        annot_transform = Identity()

    if target_id == "temporal-segmentation":
        return FrameLabelsTarget(
            segment_transform=Compose([annot_transform, RemoveOverlapping(min_gap=2)])
        )
    elif target_id == "temporal-offsets":
        return TemporalBoundaryOffsetsTarget(segment_transform=annot_transform)
    elif target_id == "segments":
        return SegmentTarget(segment_transform=annot_transform)
    elif target_id == "bio-tags":
        return BIOLabelsTarget()
    elif target_id == "boundaries-segments":
        return SegmentBoundarySegmentsTarget(
            width=2,
            segment_transform=Compose(
                [RemoveOverlapping(min_gap=0), CloseShortSilences(max_silence=16)]
            ),
        )
    elif target_id == "boundaries-segmentation":
        return SegmentBoundaryLabelsTarget(
            width=2,
            segment_transform=Compose(
                [RemoveOverlapping(min_gap=0), CloseShortSilences(max_silence=16)]
            ),
        )
    elif target_id == "boundaries-offsets":
        return SegmentBoundaryOffsetsTarget(
            width=2,
            segment_transform=Compose(
                [RemoveOverlapping(min_gap=0), CloseShortSilences(max_silence=16)]
            ),
        )
    raise ValueError(f"Unknown target: {target_id}")
