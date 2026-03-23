import torch
from torch import Tensor

from slp.decoders.base import SegmentDecoder
from slp.decoders.scoring import ProposalScorer, ActionScorer
from slp.utils.nms import soft_nms
from slp.utils.proposals import generate_proposals


class OffsetProposalDecoder(SegmentDecoder):
    """Decodes temporal offset predictions into segments via proposal
    generation and soft-NMS.

    Expects logits to contain:
        - A regression head of shape (2, T) with start/end offset predictions.
        - A classification head (consumed by the scorer).

    Args:
        regression_head: Key for the offset logits in the logits dict.
        scorer: A ProposalScorer instance. Defaults to ActionScorer.
        soft_nms_method: Soft-NMS method ("gaussian" or "linear").
        soft_nms_sigma: Gaussian sigma for soft-NMS.
        soft_nms_threshold: Score threshold to discard proposals.
    """

    def __init__(
        self,
        regression_head: str = "regression",
        scorer: ProposalScorer | None = None,
        soft_nms_method: str = "gaussian",
        soft_nms_sigma: float = 0.2,
        soft_nms_threshold: float = 0.5,
    ):
        self.regression_head = regression_head
        self.scorer = scorer or ActionScorer()
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold

    def decode(self, logits: dict[str, Tensor], n_classes: int) -> Tensor:
        device = logits[self.regression_head].device

        start_offsets, end_offsets = (
            logits[self.regression_head].detach().cpu().unbind(dim=0)
        )
        start_offsets = start_offsets.numpy()
        end_offsets = end_offsets.numpy()

        proposals, proposal_indices = generate_proposals(
            start_offsets, end_offsets, return_indices=True
        )

        if len(proposals) == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=device)

        proposal_scores = self.scorer(logits, proposal_indices)
        proposals, _ = soft_nms(
            proposals,
            scores=proposal_scores,
            method=self.soft_nms_method,
            sigma=self.soft_nms_sigma,
            threshold=self.soft_nms_threshold,
        )

        proposals = proposals.round().astype("int32")
        proposals = proposals[proposals[:, 0].argsort()]
        return torch.from_numpy(proposals).long().to(device)