import torch
from torch import Tensor
import numpy as np
from sign_language_tools.annotations.transforms import LinearBoundaryOffset

from slp.codecs.segmentation.base import SegmentationCodec
from slp.utils.nms import soft_nms
from slp.utils.proposals import generate_proposals


class OffsetsCodec(SegmentationCodec):
    def __init__(
        self,
        scoring_codec: SegmentationCodec,
        soft_nms_method: str = "gaussian",
        soft_nms_sigma: float = 0.2,
        soft_nms_threshold: float = 0.5,
    ):
        super().__init__()
        self.per_frame_codec = scoring_codec
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold

    def encode_segments_to_frame_targets(self, segments: np.ndarray, nb_frames: int) -> np.ndarray:
        assert nb_frames is not None, "Length must be provided for offsets."
        to_start_offset = LinearBoundaryOffset(
            nb_frames, ref_location="start", background_class=-1
        )
        start_offsets = to_start_offset(segments)
        to_end_offset = LinearBoundaryOffset(
            nb_frames, ref_location="end", background_class=-1
        )
        end_offsets = to_end_offset(segments)
        scores = self.per_frame_codec.encode_segments_to_frame_targets(segments, nb_frames)
        return np.stack([scores, start_offsets, end_offsets], axis=-1)

    def encode_segments_to_segment_targets(self, segments: np.ndarray) -> np.ndarray:
        return segments

    def decode_logits_to_frame_probabilities(self, logits: Tensor, n_classes: int) -> Tensor:
        return self.per_frame_codec.decode_logits_to_frame_probabilities(logits, n_classes)

    def decode_logits_to_segments(self, logits: Tensor, n_classes: int) -> Tensor:
        device = logits.device
        frame_probs = self.per_frame_codec.decode_logits_to_frame_probabilities(logits, n_classes).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        start_offsets = logits[:, n_classes]
        end_offsets = logits[:, n_classes + 1]
        proposals, proposal_indices = generate_proposals(start_offsets, end_offsets, return_indices=True)
        proposal_scores = 1 - frame_probs[proposal_indices, 0]
        proposals, _ = soft_nms(
            proposals,
            scores=proposal_scores,
            method=self.soft_nms_method,
            sigma=self.soft_nms_sigma,
            threshold=self.soft_nms_threshold,
        )
        proposals = proposals.round().astype("int32")
        return torch.from_numpy(proposals[proposals[:, 0].argsort()]).long().to(device)
