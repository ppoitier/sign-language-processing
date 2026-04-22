from typing import Optional

from torch import nn

from slp.core.config.training import TrainingConfig
from slp.utils.model import count_parameters

from slp.trainers.generic import GenericTrainer
from slp.schedulers.types import OptimizerFactory, SchedulerFactory


class IsolatedRecognitionTrainer(GenericTrainer):
    """Trainer for isolated sign recognition tasks.

    The model consumes a temporal clip (B, D, T) and internally pools to
    produce classification logits of shape (B, C). Adds user-provided
    classification metrics (train/val/test).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        heads_to_targets: dict[str, str],
        n_classes: int,
        is_output_multistage: bool = False,
        classification_head: str = "classification",
        class_target: str = "class",
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_interval: str = "epoch",
        scheduler_monitor: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            learning_rate=learning_rate,
            heads_to_targets=heads_to_targets,
            is_output_multistage=is_output_multistage,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_interval=scheduler_interval,
            scheduler_monitor=scheduler_monitor,
        )
        self.n_classes = n_classes
        self.classification_head = classification_head
        self.class_target = class_target

        self.classification_metrics = nn.ModuleDict(
            {
                "training_metrics": metrics_factory(
                    prefix="training/", n_classes=n_classes
                ),
                "validation_metrics": metrics_factory(
                    prefix="validation/", n_classes=n_classes
                ),
                "testing_metrics": metrics_factory(
                    prefix="testing/", n_classes=n_classes
                ),
            }
        )

        self.save_hyperparameters(
            ignore=[
                "model",
                "criterion",
                "test_logits",
                "metrics_factory",
                "optimizer_factory",
                "scheduler_factory",
            ]
        )

    def compute_metrics(self, logits: dict, batch: dict, mode: str) -> dict:
        cls_logits = logits[self.classification_head].detach()
        probs = cls_logits.softmax(dim=1)
        class_targets = batch["targets"][self.class_target]
        return self.classification_metrics[f"{mode}_metrics"](probs, class_targets)

    def cache_test_logits(self, logits: dict, batch: dict) -> None:
        """Store per-instance classification logits.

        Overrides the segmentation version which slices by temporal length.
        ISR logits have shape (B, C) with no temporal dimension.
        """
        instance_ids = batch["id"]
        for idx in range(len(instance_ids)):
            key = str(instance_ids[idx])
            self.test_logits[key] = {
                head_name: head_logits[idx].detach().cpu().numpy().astype("float16")
                for head_name, head_logits in logits.items()
            }


def load_isolated_recognition_trainer(
    model: nn.Module,
    criterion: nn.Module,
    training_config: TrainingConfig,
    optimizer_factory: Optional[OptimizerFactory] = None,
    scheduler_factory: Optional[SchedulerFactory] = None,
    scheduler_interval: str = "epoch",
    scheduler_monitor: Optional[str] = None,
) -> IsolatedRecognitionTrainer:
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")

    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return IsolatedRecognitionTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            criterion=criterion,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_interval=scheduler_interval,
            scheduler_monitor=scheduler_monitor,
            weights_only=False,
        )

    n_classes = training_config.n_classes
    if n_classes is None:
        raise ValueError(
            "The number of classes (n_classes) must be provided in the training configuration."
        )

    return IsolatedRecognitionTrainer(
        model=model,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        heads_to_targets=training_config.heads_to_targets,
        is_output_multistage=training_config.is_output_multistage,
        n_classes=n_classes,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        scheduler_interval=scheduler_interval,
        scheduler_monitor=scheduler_monitor,
    )