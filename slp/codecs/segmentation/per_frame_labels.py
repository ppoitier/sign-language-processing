import numpy as np
import torch
from torch import Tensor
from sign_language_tools.annotations.transforms import SegmentsToSegmentationVector, SegmentationVectorToSegments

from slp.codecs.segmentation.base import SegmentationCodec


class PerFrameLabelsCodec(SegmentationCodec):
    def __init__(self, binary: bool = True, background_label: int = 0):
        super().__init__()
        self.binary = binary
        self.background_label = background_label
        self.to_segments = SegmentationVectorToSegments(
            background_classes=(background_label,),
            use_annotation_labels=not binary,
        )

    def encode_segments_to_frame_targets(self, segments: np.ndarray, nb_frames: int) -> np.ndarray:
        to_frame_labels = SegmentsToSegmentationVector(
            vector_size=nb_frames,
            background_label=self.background_label,
            use_annotation_labels=not self.binary,
        )
        return to_frame_labels(segments)

    def encode_segments_to_segment_targets(self, segments: np.ndarray) -> np.ndarray:
        return segments

    def decode_logits_to_frame_probabilities(self, logits: Tensor, n_classes: int) -> Tensor:
        return logits[..., :n_classes].softmax(dim=-1)

    def decode_logits_to_segments(self, logits: Tensor, n_classes: int) -> Tensor:
        return torch.from_numpy(self.to_segments(logits[..., :n_classes].argmax(dim=-1).detach().cpu().numpy())).long().to(logits.device)
