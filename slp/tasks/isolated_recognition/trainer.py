from torch import nn, optim

from slp.config.templates.training import TrainingConfig
from slp.metrics.classification.base import ClassificationMetrics
from slp.trainers.base import TrainerBase
from slp.utils.model import count_parameters
from slp.nn.pose_transformer import PoseTransformer


class IsolatedRecognitionTrainer(TrainerBase):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        n_classes: int,
        is_output_multilayer: bool,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.is_output_multilayer = is_output_multilayer

        self.training_metrics = ClassificationMetrics(prefix="training/cls/", n_classes=n_classes)
        self.validation_metrics = ClassificationMetrics(prefix="validation/cls/", n_classes=n_classes)
        self.testing_metrics = ClassificationMetrics(prefix="testing/cls/", n_classes=n_classes)

        self.test_logits = {}
        self.save_hyperparameters(ignore=["model", "criterion", "test_logits"])

    def prediction_step(self, batch, mode):
        features, masks, targets = batch["poses"].float(), batch["masks"].bool(), batch["label_id"].long()
        # features, masks, targets = batch["video"].float(), batch["masks"].bool(), batch["label_id"].long()
        # features of shape (N, C_in, T)
        # masks of shape (N, 1, T)
        # targets of shape N
        batch_size = features.size(0)
        logits = self.model(features, masks)
        loss = self.criterion(logits, {'classification': targets})

        self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        cls_logits = logits["classification"]
        # cls_logits = logits["classification"].max(dim=2)[0]
        # classification logits of shape (N, C_out)
        cls_logits = cls_logits[-1] if self.is_output_multilayer else cls_logits
        probs = cls_logits.softmax(dim=-1)
        # probabilities of shape (N, C_out)
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
        instance_ids, lengths, targets = batch["id"], batch["length"], batch["label"]
        logits, loss, metrics = self.prediction_step(batch, "testing")
        ...  # todo

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer


def load_isolated_recognition_trainer(
    model: nn.Module,
    criterion: nn.Module,
    training_config: TrainingConfig,
):
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")
    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return IsolatedRecognitionTrainer.load_from_checkpoint(
            checkpoint_path, model=model, criterion=criterion
        )
    n_classes = training_config.n_classes
    if n_classes is None:
        raise ValueError("The number of classes (n_classes) must be provided in the training configuration.")
    return IsolatedRecognitionTrainer(
        model=model,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        n_classes=n_classes,
        is_output_multilayer=training_config.is_output_multilayer,
    )
