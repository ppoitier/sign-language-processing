import numpy as np
import torch
from torch import Tensor

from sign_language_tools.annotations.transforms import (
    SegmentsToBoundaries,
    SegmentationVectorToSegments,
    FillBetween,
    RemoveOverlapping,
    MergeSegments,
)
from sign_language_tools.common.transforms import Compose
from slp.codecs.segmentation.base import SegmentationCodec
from slp.codecs.segmentation.per_frame_labels import PerFrameLabelsCodec


class BoundariesCodec(SegmentationCodec):
    def __init__(
        self,
        width: int = 2,
        min_gap: int = 0,
        filled_gap_max_duration: int = 8,
    ):
        super().__init__()
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

    def encode_segments_to_frame_targets(
        self, segments: np.ndarray, nb_frames: int
    ) -> np.ndarray:
        per_frame_codec = PerFrameLabelsCodec(binary=True, background_label=0)
        return per_frame_codec.encode_segments_to_frame_targets(
            self.encode_segments_to_segment_targets(segments), nb_frames
        )

    def encode_segments_to_segment_targets(self, segments: np.ndarray) -> np.ndarray:
        return self.from_segments_to_boundaries(segments)[:, :2]

    def decode_logits_to_frame_probabilities(
        self, logits: Tensor, n_classes: int
    ) -> Tensor:
        return logits[..., :n_classes].softmax(dim=-1)

    def decode_logits_to_segments(self, logits: Tensor, n_classes: int) -> Tensor:
        return (
            torch.from_numpy(
                self.from_logits_to_segments(
                    logits.argmax(dim=-1).detach().cpu().numpy()
                )
            )
            .long()
            .to(logits.device)
        )
