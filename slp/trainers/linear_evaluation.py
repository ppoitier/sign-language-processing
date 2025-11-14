import torch
from torch import nn, optim

from slp.config.templates.training import TrainingConfig
from slp.metrics.classification.base import ClassificationMetrics
from slp.trainers.base import TrainerBase
from slp.utils.model import count_parameters
from slp.schedulers.cosine_annealing_with_linear_warmup import create_warmup_plateau_cosine_scheduler


class LinearEvaluationTrainer(TrainerBase):

    def __init__(
        self,
        backbone: nn.Module,
        cls_head: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        n_epochs: int,
        n_warmup_epochs: int,
        n_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.criterion = criterion

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.n_classes = n_classes

        self.training_metrics = ClassificationMetrics(prefix="training/cls/", n_classes=n_classes)
        self.validation_metrics = ClassificationMetrics(prefix="validation/cls/", n_classes=n_classes)
        self.testing_metrics = ClassificationMetrics(prefix="testing/cls/", n_classes=n_classes)

        self.test_logits = {}
        self.save_hyperparameters(ignore=["backbone", "cls_head", "criterion", "test_logits"])

    def prediction_step(self, batch, mode):
        features, masks, targets = batch["poses"].float(), batch["masks"].bool(), batch["label_id"].long()
        batch_size = features.size(0)
        with torch.no_grad():
            embeddings = self.backbone(features, masks)
        logits = self.cls_head(embeddings.detach())
        loss = self.criterion(logits, targets)
        self.log(f"{mode}/cls_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        probs = logits.softmax(dim=-1)
        if mode == "training":
            metrics = self.training_metrics(probs, targets)
        elif mode == "validation":
            metrics = self.validation_metrics(probs, targets)
        elif mode == "testing":
            metrics = self.testing_metrics(probs, targets)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.log_metrics(metrics, batch_size=batch_size)
        return logits, loss, metrics

    def training_step(self, batch, batch_idx):
        _, loss, _ = self.prediction_step(batch, "training")
        return loss

    def validation_step(self, batch, batch_index):
        self.prediction_step(batch, mode="validation")

    def test_step(self, batch, batch_index):
        logits, loss, metrics = self.prediction_step(batch, "testing")
        ...  # todo

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = create_warmup_plateau_cosine_scheduler(
            optimizer=optimizer,
            n_warmup_steps=self.n_warmup_epochs,
            n_plateau_steps=0,
            max_steps=self.n_epochs,
            lr=self.learning_rate,
            start_lr=1e-6,
            end_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def load_linear_evaluation_trainer(
    backbone: nn.Module,
    cls_head: nn.Module,
    criterion: nn.Module,
    training_config: TrainingConfig,
):
    print(f"Total number of parameters in the backbone model: {count_parameters(backbone):,}")
    print(f"Total number of parameters in the classification head model: {count_parameters(cls_head):,}")
    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return LinearEvaluationTrainer.load_from_checkpoint(checkpoint_path, backbone=backbone, cls_head=cls_head, criterion=criterion)
    n_classes = training_config.n_classes
    if n_classes is None:
        raise ValueError("The number of classes (n_classes) must be provided in the training configuration.")
    return LinearEvaluationTrainer(
        backbone=backbone,
        cls_head=cls_head,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        n_epochs=training_config.max_epochs,
        n_warmup_epochs=training_config.n_warmup_epochs,
        n_classes=n_classes,
    )
