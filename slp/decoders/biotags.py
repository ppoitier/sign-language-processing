import torch
from torch import Tensor

from sign_language_tools.annotations.transforms import SegmentationVectorToSegments

from slp.core.registry import SEGMENT_DECODER_REGISTRY
from slp.decoders.base import SegmentDecoder


@SEGMENT_DECODER_REGISTRY.register("biotags-argmax")
class BioTagsDecoder(SegmentDecoder):

    def __init__(
        self,
        classification_head: str = "classification",
        b_tag_class: int = 2,
        i_tag_class: int = 1,
    ):
        self.classification_head = classification_head
        self.b_tag_class = b_tag_class
        self.i_tag_class = i_tag_class
        self.to_segments = SegmentationVectorToSegments(
            background_classes=(0,),
            use_annotation_labels=False,
        )

    def decode(self, logits: dict[str, Tensor], n_classes: int) -> Tensor:
        preds = logits[self.classification_head].argmax(dim=0).detach()
        binary_segmentation = ((preds == self.b_tag_class) | (preds == self.i_tag_class)).long()
        segments = self.to_segments(binary_segmentation.cpu().numpy())
        return torch.from_numpy(segments).to(preds.device).long()
