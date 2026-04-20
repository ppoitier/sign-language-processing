from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from sign_language_tools.annotations.transforms import (
    ScaleSegments,
    SegmentsToSegmentationVector,
)
from sldl.targets.target import TargetEncoder


class BIOSegmentation(TargetEncoder):
    def __init__(
        self,
        annotation_id="both_hands",
        begin_width: float | int = 0.3,
        pad_value: int = -100,
    ):
        super().__init__()
        self.annotation_id = annotation_id
        self.to_b_segments = ScaleSegments(factor=begin_width, location="start")
        self.to_i_segments = ScaleSegments(factor=1.0 - begin_width, location="end")
        self.pad_value = pad_value

    def encode(self, sample: dict) -> Any:
        segments = (
            sample["annotations"][self.annotation_id]
            .loc[:, ["start_frame", "end_frame"]]
            .values
        )
        to_temporal_segmentation = SegmentsToSegmentationVector(
            vector_size=sample["n_frames"],
            use_annotation_labels=False,
            background_label=0,
            fill_label=1,
        )
        b_tags = to_temporal_segmentation(self.to_b_segments(segments))
        i_tags = to_temporal_segmentation(self.to_i_segments(segments))
        b_indices = b_tags > 0
        i_tags[b_indices] = 2
        return torch.from_numpy(i_tags)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        )
