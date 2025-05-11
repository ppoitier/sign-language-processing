from torchmetrics import Metric
import torch
from torch import Tensor

from .functional import compute_iou_matrix


class MeanMaxIoUOverSegments(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("accumulator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_segments: list[Tensor], gt_segments: list[Tensor]) -> None:
        """
        Args:
            pred_segments: batch of tensors with shape (N, 2) for the start and the end of N predicted segments.
            gt_segments: batch of tensors with shape (N, 2) for the start and the end of N predicted segments.
        """

        for pred_segments_b, gt_segments_b in zip(pred_segments, gt_segments):
            num_preds, num_gt = len(pred_segments_b), len(gt_segments_b)
            # Edge cases
            if num_gt == 0:
                continue
            if num_preds == 0:
                self.count += num_gt
                continue
            max_ious, _ = compute_iou_matrix(gt_segments_b, pred_segments_b).max(dim=1)
            self.accumulator += max_ious.sum()
            self.count += max_ious.numel()

    def compute(self) -> Tensor:
        if self.count == 0:
            return torch.tensor(0.0, device=self.count.device)
        return self.accumulator / self.count
