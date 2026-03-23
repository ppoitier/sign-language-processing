from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import rv_continuous
from torch import Tensor


class ProposalScorer(ABC):
    """Scores proposal segments based on model logits."""

    @abstractmethod
    def __call__(self, logits: dict[str, Tensor], proposal_indices: np.ndarray) -> np.ndarray:
        ...


class ActionScorer(ProposalScorer):
    """Scores proposals by (1 - background probability) at proposal frames."""

    def __init__(self, classification_head: str = "classification", background_class: int = 0):
        self.classification_head = classification_head
        self.background_class = background_class

    def __call__(self, logits: dict[str, Tensor], proposal_indices: np.ndarray) -> np.ndarray:
        cls_probs = logits[self.classification_head].softmax(dim=0).detach().cpu().numpy()
        return 1 - cls_probs[self.background_class, proposal_indices]


class ActionWithDurationScorer(ProposalScorer):
    """Scores proposals using both classification confidence and a duration prior."""

    def __init__(
        self,
        duration_distribution: rv_continuous,
        coef: float = 1.0,
        classification_head: str = "classification",
        regression_head: str = "regression",
        background_class: int = 0,
    ):
        self.duration_distribution = duration_distribution
        self.coef = coef
        self.classification_head = classification_head
        self.regression_head = regression_head
        self.background_class = background_class

    def __call__(self, logits: dict[str, Tensor], proposal_indices: np.ndarray) -> np.ndarray:
        cls_probs = logits[self.classification_head].softmax(dim=0).detach().cpu().numpy()
        proposal_scores = 1 - cls_probs[self.background_class, proposal_indices]

        offsets = logits[self.regression_head].detach().cpu().numpy()[:, proposal_indices]
        durations = offsets[1, :] - offsets[0, :] + 1
        duration_likelihoods = self.duration_distribution.pdf(durations)

        return proposal_scores + self.coef * duration_likelihoods