from abc import ABC
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from sldl.targets.target import TargetEncoder
from sign_language_tools.core.transform import Transform
from sign_language_tools.annotations.transforms import (
    SegmentsToBoundaries,
    SegmentsToFrameLabels,
    SegmentsToBoundaryOffsets,
)


class _BoundarySegmentsBase(TargetEncoder, ABC):
    """Shared pipeline for boundary-based targets: annotations → boundary segments.

    Not meant to be used directly. Subclasses implement `encode` and `collate`
    to render the boundary segments into their desired output shape.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        width: int | None = 4,
        relative_width: float | None = None,
        min_width: int = 2,
        boundary_labels: tuple[int, int] = (1, 2),
        segment_transform: Transform | None = None,
    ):
        super().__init__()
        self.annotation_id = annotation_id
        self.segment_transform = segment_transform
        self._to_boundaries = SegmentsToBoundaries(
            width=width,
            relative_width=relative_width,
            min_width=min_width,
            boundary_labels=boundary_labels,
        )

    def _boundary_segments(self, sample: dict) -> tuple[np.ndarray, int]:
        n_frames = sample.get("n_frames", 0)
        annotations = sample.get("annotations", {}).get(self.annotation_id)

        if annotations is None or annotations.empty:
            return np.zeros((0, 3), dtype=np.int64), n_frames

        segments = annotations[["start_frame", "end_frame"]].to_numpy()
        if self.segment_transform is not None:
            segments = self.segment_transform(segments)
        return self._to_boundaries(segments), n_frames


class SegmentBoundarySegmentsTarget(_BoundarySegmentsBase):
    """Raw boundary segments as `(N, 3)` int tensor: [b_start, b_end, label].

    Variable-length output — collated into a padded `(batch, max_N, 3)`
    tensor. Use this when the model consumes segment-level annotations
    directly (e.g. a transformer that attends over boundary tokens).
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        width: int | None = 4,
        relative_width: float | None = None,
        min_width: int = 2,
        boundary_labels: tuple[int, int] = (1, 2),
        pad_value: int = -1,
        segment_transform: Transform | None = None,
    ):
        super().__init__(
            annotation_id=annotation_id,
            width=width,
            relative_width=relative_width,
            min_width=min_width,
            boundary_labels=boundary_labels,
            segment_transform=segment_transform,
        )
        self.pad_value = pad_value

    def encode(self, sample: dict) -> Any:
        boundaries, _ = self._boundary_segments(sample)
        return torch.from_numpy(boundaries.astype(np.int64))

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        )


class SegmentBoundaryLabelsTarget(_BoundarySegmentsBase):
    """Per-frame classification target marking boundary regions.

    Output: `(time,)` int tensor where each frame carries the boundary
    label from `boundary_labels` (start or end) or `background_id`.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        width: int | None = 4,
        relative_width: float | None = None,
        min_width: int = 2,
        boundary_labels: tuple[int, int] = (1, 2),
        background_id: int = 0,
        pad_value: int = -100,
        segment_transform: Transform | None = None,
    ):
        super().__init__(
            annotation_id=annotation_id,
            width=width,
            relative_width=relative_width,
            min_width=min_width,
            boundary_labels=boundary_labels,
            segment_transform=segment_transform,
        )
        self.background_id = background_id
        self.pad_value = pad_value
        self._renderer = SegmentsToFrameLabels(background_label=background_id)

    def encode(self, sample: dict) -> Any:
        boundaries, n_frames = self._boundary_segments(sample)
        labels = self._renderer(boundaries, vector_size=n_frames)
        return torch.from_numpy(labels)

    def collate(self, batch_targets: list[Any]) -> Any:
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        )


class SegmentBoundaryOffsetsTarget(_BoundarySegmentsBase):
    """Per-frame regression target: distance to the current boundary region's edges.

    Output (after collate): `(batch, 2, time)` float tensor. Channel 0 is
    `start_offset`, channel 1 is `end_offset`. Frames outside any boundary
    region get `background_value`.
    """

    def __init__(
        self,
        annotation_id: str = "both_hands",
        width: int | None = 4,
        relative_width: float | None = None,
        min_width: int = 2,
        background_value: float = -1.0,
        pad_value: float = -1.0,
        segment_transform: Transform | None = None,
    ):
        super().__init__(
            annotation_id=annotation_id,
            width=width,
            relative_width=relative_width,
            min_width=min_width,
            boundary_labels=(1, 1),  # unused; offsets ignore labels
            segment_transform=segment_transform,
        )
        self.background_value = background_value
        self.pad_value = pad_value
        self._renderer = SegmentsToBoundaryOffsets(
            background_value=background_value,
        )

    def encode(self, sample: dict) -> Any:
        boundaries, n_frames = self._boundary_segments(sample)
        offsets = self._renderer(boundaries, sequence_length=n_frames)
        return torch.from_numpy(offsets)

    def collate(self, batch_targets: list[Any]) -> Any:
        # (batch, time, 2) -> (batch, 2, time) for channel-first models.
        return pad_sequence(
            batch_targets, batch_first=True, padding_value=self.pad_value
        ).permute(0, 2, 1)
