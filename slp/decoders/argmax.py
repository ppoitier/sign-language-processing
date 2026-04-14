import torch
from torch import Tensor

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

    def decode(self, logits: dict[str, Tensor], n_classes: int) -> Tensor:
        device = logits[self.classification_head].device
        preds = logits[self.classification_head].argmax(dim=0)  # (T,)
        mask = preds != self.background_class

        if not mask.any():
            return torch.zeros((0, 2), dtype=torch.long, device=device)

        # Find boundaries where mask changes
        diff = torch.diff(mask.long(), prepend=torch.tensor([0], device=device))
        starts = (diff == 1).nonzero(as_tuple=False).squeeze(-1)
        ends = (diff == -1).nonzero(as_tuple=False).squeeze(-1)

        # Handle case where last segment runs to the end
        if mask[-1]:
            ends = torch.cat([ends, torch.tensor([len(preds)], device=device)])

        return torch.stack([starts, ends], dim=1).long()