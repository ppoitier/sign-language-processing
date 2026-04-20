import torch
from torch import Tensor

from sign_language_tools.annotations.transforms import SegmentationVectorToSegments

from slp.core.registry import SEGMENT_DECODER_REGISTRY
from slp.decoders.base import SegmentDecoder


@SEGMENT_DECODER_REGISTRY.register("argmax")
class ArgmaxDecoder(SegmentDecoder):
    """Decodes frame-level classification logits into segments by taking
    the argmax and grouping contiguous runs of non-background classes.

    Returns an (S, 2) tensor of [start_frame, end_frame] pairs.

    Args:
        classification_head: Key for the classification logits.
        background_class: Class index treated as background (excluded from segments).
    """

    def __init__(
        self,
        classification_head: str = "classification",
        background_class: int = 0,
    ):
        self.classification_head = classification_head
        self.background_class = background_class
        self.to_segments = SegmentationVectorToSegments(
            background_classes=(background_class,), use_annotation_labels=False
        )

    def decode(self, logits: dict[str, Tensor], n_classes: int) -> Tensor:
        preds = logits[self.classification_head].argmax(dim=0).detach()
        segments = self.to_segments(preds.cpu().numpy())
        return torch.from_numpy(segments).to(preds.device).long()
