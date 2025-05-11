import torch
from torch import Tensor
from torchmetrics import Metric

from .functional import tp_fp_fn


class MeanIoUF1Score(Metric):
    def __init__(self, thresholds: Tensor, **kwargs):
        super().__init__(**kwargs)
        self.add_state("thresholds", default=thresholds, persistent=True)
        self.add_state(
            "tp", default=torch.zeros(thresholds.size(0)), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp", default=torch.zeros(thresholds.size(0)), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn", default=torch.zeros(thresholds.size(0)), dist_reduce_fx="sum"
        )

    def update(self, pred_segments: list[Tensor], gt_segments: list[Tensor]) -> None:
        """
        Args:
            pred_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
            gt_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
        """
        for pred_segments_b, gt_segments_b in zip(pred_segments, gt_segments):
            tp, fp, fn = tp_fp_fn(pred_segments_b, gt_segments_b, self.thresholds)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def compute(self) -> Tensor:
        tp, fp, fn = self.tp, self.fp, self.fn
        denom_prec = tp + fp
        precisions = torch.where(denom_prec > 0, tp / denom_prec, torch.zeros_like(tp))
        denom_rec = tp + fn
        recalls = torch.where(denom_rec > 0, tp / denom_rec, torch.zeros_like(tp))
        denom_f1 = precisions + recalls
        f1_scores = torch.where(
            denom_f1 > 0,
            2 * precisions * recalls / denom_f1,
            torch.zeros_like(denom_f1),
        )
        return f1_scores.mean()
