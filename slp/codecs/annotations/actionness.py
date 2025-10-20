from typing import Union

import numpy as np
import torch
from torch import Tensor

from slp.codecs.annotations.base import AnnotationCodec
from sign_language_tools.annotations.transforms import SegmentsToSegmentationVector, SegmentationVectorToSegments


class ActionnessCodec(AnnotationCodec):
    def __init__(self, background_label: int = 0, action_label: int = 1):
        super().__init__()
        self.background_label = background_label
        self.action_label = action_label

        self.to_segments = SegmentationVectorToSegments(
            background_classes=(background_label,),
            use_annotation_labels=False,
        )

    def encode(self, annotations: np.ndarray, n_frames: int):
        """
        Args:
            annotations: Array of shape (M, 3) for M annotations with a start, end and a label.
                         The time unit must be in frames !
            n_frames: Total number of frames in the resulting sequences.

        Returns:
            per_frame_sequence: Array of shape T that contains the activity of the signer for each frame.
        """
        to_frame_labels = SegmentsToSegmentationVector(
            vector_size=n_frames,
            background_label=self.background_label,
            fill_label=self.action_label,
            use_annotation_labels=False,
        )
        return {
            'segments': annotations[:, :2],
            'frame_labels': to_frame_labels(annotations).astype('int64'),
        }

    def decode(self, logits: dict[str, Union[Tensor, np.ndarray]], n_classes: int) -> Tensor:
        """
        Args:
            logits: Dict containing an array (key=classification) of shape (N, C, T) containing the output of
                    a frame-classification model. N is the batch size, C is the number of classes, and T is the number
                    of frames.
            n_classes: Number of classes.
        Returns:
            segments: Decoded segments from the logits.
        """
        preds = logits['classification']
        if isinstance(preds, np.ndarray):
            device = 'cpu'
            preds = torch.from_numpy(preds).argmax(0)
        else:
            device = preds.device
            preds = preds.detach().argmax(0).cpu()
        return torch.from_numpy(self.to_segments(preds.numpy())).long().to(device)
