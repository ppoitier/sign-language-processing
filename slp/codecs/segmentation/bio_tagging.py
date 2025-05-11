import numpy as np
import torch
from torch import Tensor

from sign_language_tools.annotations.transforms import BioTags
from slp.codecs.segmentation.base import SegmentationCodec
from slp.codecs.segmentation.per_frame_labels import PerFrameLabelsCodec
from slp.utils.bio_tags import bio_probabilities_to_segments


class BioTaggingCodec(SegmentationCodec):
    def __init__(self, b_tag_size: int | float = 0.25):
        super().__init__()
        self.from_segments_to_bio_segments = BioTags(b_tag_size=b_tag_size)
        self.per_frame_codec = PerFrameLabelsCodec(binary=False, background_label=0)

    def encode_segments_to_frame_targets(
        self, segments: np.ndarray, nb_frames: int
    ) -> np.ndarray:
        return self.per_frame_codec.encode_segments_to_frame_targets(
            self.from_segments_to_bio_segments(segments), nb_frames
        )

    def encode_segments_to_segment_targets(self, segments: np.ndarray) -> np.ndarray:
        # return self.from_segments_to_bio_segments(segments)
        return segments

    def decode_logits_to_frame_probabilities(
        self, logits: Tensor, n_classes: int
    ) -> Tensor:
        return logits[..., :n_classes].softmax(dim=-1)

    def decode_logits_to_segments(self, logits: Tensor, n_classes: int) -> Tensor:
        bio_probs = (
            self.decode_logits_to_frame_probabilities(logits, n_classes)
            .detach()
            .cpu()
            .numpy()
        )
        return (
            torch.from_numpy(
                bio_probabilities_to_segments(
                    bio_probs, threshold_b=0.5, threshold_o=0.5
                )
            )
            .long()
            .to(logits.device)
        )
