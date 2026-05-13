from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from sign_language_tools.annotations.transforms import (
    BioTags,
    SegmentsToFrameLabels,
)
from sldl.targets.target import TargetEncoder


class BIOLabelsTarget(TargetEncoder):
    def __init__(
        self,
        annotation_id="both_hands",
        relative_width: float | None = 0.3,
        width: int | None = None,
        pad_value: int = -100,
    ):
        super().__init__()
        self.annotation_id = annotation_id
        self.pad_value = pad_value
        self._to_bio_segments = BioTags(width=width, relative_width=relative_width)

    def encode(self, sample: dict) -> Any:
        segments = (
            sample["annotations"][self.annotation_id]
            .loc[:, ["start_frame", "end_frame"]]
            .values
        )
        bio_segments = self._to_bio_segments(segments)
        to_frame_labels = SegmentsToFrameLabels(vector_size=sample["n_frames"])
        bio_tags = to_frame_labels(bio_segments)
        return torch.from_numpy(bio_tags)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        )
