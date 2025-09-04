import numpy as np

from sign_language_tools.annotations.transforms import (
    SegmentsToBoundaries,
    FillBetween,
    MergeSegments,
    RemoveOverlapping,
    SegmentationVectorToSegments,
    SegmentsToSegmentationVector,
)
from sign_language_tools.common.transforms import Compose
from slp.codecs.annotations.base import AnnotationCodec


class BoundariesCodec(AnnotationCodec):

    def __init__(
        self,
        width: int = 2,
        min_gap: int = 0,
        filled_gap_max_duration: int = 8,
        background_label: int = 0,
        action_label: int = 1,
    ):
        super().__init__()
        self.background_label = background_label
        self.action_label = action_label
        self.from_segments_to_boundaries = Compose(
            [
                SegmentsToBoundaries(width=width, boundary_label=(1, 2)),
                FillBetween(
                    start_value=2,
                    end_value=1,
                    fill_value=1,
                    max_width=filled_gap_max_duration,
                ),
                MergeSegments(),
                RemoveOverlapping(min_gap=min_gap),
            ]
        )
        self.from_logits_to_segments = SegmentationVectorToSegments(
            background_classes=(-1, 0),
            use_annotation_labels=False,
        )

    def encode(self, annotations: np.ndarray, n_frames: int):
        boundary_segments = self.from_segments_to_boundaries(annotations)
        to_frame_labels = SegmentsToSegmentationVector(
            vector_size=n_frames,
            background_label=self.background_label,
            fill_label=self.action_label,
            use_annotation_labels=False,
        )
        return {
            'segments': boundary_segments,
            'frame_labels': to_frame_labels(boundary_segments),
        }
