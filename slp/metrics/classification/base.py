from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score


class ClassificationMetrics(MetricCollection):
    def __init__(self, n_classes: int, **kwargs):
        acc_args = dict(task="multiclass", num_classes=n_classes, ignore_index=-1)
        metrics = {
            "macro_accuracy": Accuracy(average="macro", **acc_args),
            "micro_accuracy": Accuracy(average="micro", **acc_args),
            "top_3_accuracy": Accuracy(top_k=3, average="micro", **acc_args),
            "top_5_accuracy": Accuracy(top_k=5, average="micro", **acc_args),
            "top_10_accuracy": Accuracy(top_k=10, average="micro", **acc_args),
            "recall": Recall(average=None, **acc_args),
            "precision": Precision(average=None, **acc_args),
            "f1": F1Score(average='macro', **acc_args),
        }
        super().__init__(metrics, **kwargs)
