from typing import Union

import numpy as np
import torch
from torch import Tensor

from sign_language_tools.annotations.transforms import LinearBoundaryOffset
from slp.codecs.annotations.base import AnnotationCodec
from slp.utils.nms import soft_nms
from slp.utils.proposals import generate_proposals


class OffsetsCodec(AnnotationCodec):
    def __init__(
        self,
        soft_nms_method: str = "gaussian",
        soft_nms_sigma: float = 0.2,
        soft_nms_threshold: float = 0.5,
    ):
        super().__init__()
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold

    def encode(self, annotations: np.ndarray, n_frames: int):
        to_start_offset = LinearBoundaryOffset(n_frames, ref_location="start", background_class=-1)
        to_end_offset = LinearBoundaryOffset(n_frames, ref_location="end", background_class=-1)
        start_offsets = to_start_offset(annotations)
        end_offsets = to_end_offset(annotations)
        return {
            'offsets': np.stack([start_offsets, end_offsets], axis=-1).astype('float32'),
        }

    def decode(self, logits: dict[str, Union[Tensor, np.ndarray]], n_classes: int) -> Tensor:
        """

        Args:
            logits: Dictionary containing outputs (tensors) of a segmentation model.
                    The key 'classification' must contain classification logits of shape (N, C, T) where
                    the class 0 represents the background.
                    The key 'regression' must contain logits of shape (N, 2, T) for, respectively, the start and
                    end offsets predictions.
            n_classes: The number of classes for the classification task.

        Shapes:
            classification: (N, C, T)
            offsets: (N, 2, T)
            where N is the batch size, C is the number of predicted classes, and T is the number of frames.

        Returns:
            proposals: predicted segments from the combination of the predicted offsets with a Soft-NMS algorithm using
            the classification scores.
        """
        logits = {
            k: v if isinstance(v, Tensor) else torch.from_numpy(v)
            for k, v in logits.items()
        }

        device = logits['regression'].device
        start_offsets, end_offsets = logits['regression'].detach().cpu().unbind(dim=0)
        start_offsets, end_offsets = start_offsets.numpy(), end_offsets.numpy()
        cls_probs = logits['classification'].softmax(dim=0).detach().cpu().numpy()

        proposals, proposal_indices = generate_proposals(start_offsets, end_offsets, return_indices=True)

        proposal_scores = 1 - cls_probs[0, proposal_indices]
        proposals, _ = soft_nms(
            proposals,
            scores=proposal_scores,
            method=self.soft_nms_method,
            sigma=self.soft_nms_sigma,
            threshold=self.soft_nms_threshold,
        )

        proposals = proposals.round().astype("int32")
        proposals = proposals[proposals[:, 0].argsort()]
        proposals = torch.from_numpy(proposals).long().to(device)
        return proposals
