from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor


class SegmentationCodec(ABC):
    @abstractmethod
    def encode_segments_to_frame_targets(self, segments: np.ndarray, nb_frames: int) -> np.ndarray:
        raise NotImplementedError("Segment encoding is not implemented.")

    @abstractmethod
    def encode_segments_to_segment_targets(self, segments: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Segment encoding is not implemented.")

    def batch_encode_segments_to_frame_targets(self, segments_batch: list[np.ndarray], nb_frames: int) -> np.ndarray:
        return np.stack([self.encode_segments_to_frame_targets(segments, nb_frames) for segments in segments_batch])

    @abstractmethod
    def decode_logits_to_frame_probabilities(self, logits: Tensor, n_classes: int) -> Tensor:
        raise NotImplementedError("Frame-label decoding is not implemented.")

    @abstractmethod
    def decode_logits_to_segments(self, logits: Tensor, n_classes: int) -> Tensor:
        raise NotImplementedError("Segment decoding is not implemented.")

    def batch_decode_logits_to_segments(self, logits_batch: Tensor, n_classes: int) -> list[Tensor]:
        return [self.decode_logits_to_segments(logits, n_classes) for logits in logits_batch.unbind(dim=0)]
