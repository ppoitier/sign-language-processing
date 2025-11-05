from torch import Tensor
from tqdm import tqdm
import torchmetrics


def compute_global_metric(
        metric: torchmetrics.Metric,
        preds: list[Tensor] | Tensor,
        targets: list[Tensor] | Tensor,
):
    """
    Computes a metric from a list of pairs of predictions and targets.
    Make sure that the metric and the tensors are on the same device.

    Args:
        metric: The metric instance to compute.
        preds: The prediction tensors (stacked or as a list).
        targets: The ground truth tensors (stacked or as a list).

    Returns:
        any: The computed metric.
    """
    metric.reset()
    metric.update(preds, targets)
    return metric.compute()


def compute_individual_metrics(
        metric: torchmetrics.Metric,
        preds: list[Tensor] | Tensor,
        targets: list[Tensor] | Tensor,
        batch_as_list: bool = False,
):
    """
    Computes individual metrics from a list of pairs of predictions and targets.
    Make sure that the metric and the tensors are on the same device.

    Args:
        metric: The metric instance to compute.
        preds: The prediction tensors (stacked or as a list).
        targets: The ground truth tensors (stacked or as a list).
        batch_as_list: Whether the single-sample batch used in the metric is a list (if True) or a tensor (otherwise). Default to false.

    Returns:
        list[any]: The computed metrics for each pair of prediction and target.
    """

    individual_metrics = []
    for pred, target in tqdm(zip(preds, targets), total=len(preds)):
        if batch_as_list:
            pred, target = [pred], [target]
        else:
            pred, target = pred.unsqueeze(0), target.unsqueeze(0)
        metric.reset()
        metric.update(pred, target)
        individual_metrics.append(metric.compute())
    return individual_metrics