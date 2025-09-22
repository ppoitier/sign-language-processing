from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score


class ClassificationMetrics(MetricCollection):
    def __init__(self, n_classes: int, **kwargs):
        acc_args = dict(task="multiclass", num_classes=n_classes, ignore_index=-1)
        metrics = {
            "macro_accuracy": Accuracy(average="macro", **acc_args),
            "micro_accuracy": Accuracy(average="micro", **acc_args),
            "macro_accuracy/top5": Accuracy(top_k=5, average="macro", **acc_args),
            "macro_accuracy/top10": Accuracy(top_k=10, average="macro", **acc_args),
            "micro_accuracy/top5": Accuracy(top_k=5, average="micro", **acc_args),
            "micro_accuracy/top10": Accuracy(top_k=10, average="micro", **acc_args),
            "f1": F1Score(average="macro", **acc_args),
        }
        super().__init__(metrics, **kwargs)
