from typing import Callable, Optional

from torch import nn, optim, Tensor

from slp.trainers.base import TrainerBase
from slp.schedulers.types import OptimizerFactory, SchedulerFactory
from slp.utils.model import count_parameters


MomentumScheduler = Callable[[int, int], float]
"""Signature: (current_step, max_steps) -> momentum coefficient in [0, 1]."""


class PretrainingTrainer(TrainerBase):
    """Trainer for self-supervised pretraining (SimCLR, SimSiam, BYOL, MoCo, DINO, ...).

    Assumes the model is a bundled SSL module (encoder + projector + optional
    predictor + optional momentum target) that consumes a list of augmented
    views and returns whatever representation structure the criterion expects.

    The trainer is deliberately agnostic to the SSL method: all method-specific
    logic (stop-gradient, asymmetric prediction, target network forward pass)
    lives inside the model and criterion. The trainer only handles:
        - routing views to the model
        - loss computation and logging
        - optional EMA target-network updates
        - optimizer/scheduler setup

    The model may optionally expose:
        - ``update_target_network(momentum: float) -> None`` for EMA methods.
          Called after each training step if ``momentum_scheduler`` is provided.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        views_key: str = "views",
        masks_key: Optional[str] = "masks",
        momentum_scheduler: Optional[MomentumScheduler] = None,
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_interval: str = "epoch",
        scheduler_monitor: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.views_key = views_key
        self.masks_key = masks_key

        self.momentum_scheduler = momentum_scheduler

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.scheduler_interval = scheduler_interval
        self.scheduler_monitor = scheduler_monitor

        self.save_hyperparameters(
            ignore=[
                "model",
                "criterion",
                "momentum_scheduler",
                "optimizer_factory",
                "scheduler_factory",
            ]
        )

    def _prepare_views(self, batch: dict) -> tuple[list[Tensor], Optional[list[Tensor]], int]:
        """Extract and reshape views (and optionally masks) into the model's input format.

        Returns:
            views: list of tensors, each (B, D, T).
            masks: list of (B, 1, T) bool tensors, or None if masks are not provided.
            batch_size: int.
        """
        raw_views = batch[self.views_key]
        batch_size = raw_views[0].size(0)

        views = [v.permute(0, 2, 1).float().contiguous() for v in raw_views]

        masks = None
        if self.masks_key is not None and self.masks_key in batch:
            raw_masks = batch[self.masks_key]
            masks = [m.unsqueeze(1).bool().contiguous() for m in raw_masks]

        return views, masks, batch_size

    def forward_step(self, batch: dict) -> tuple[Tensor, dict, int]:
        """Run the SSL model forward and compute the pretraining loss.

        Returns:
            loss: scalar total loss.
            loss_components: dict of component_name -> value.
            batch_size: int.
        """
        views, masks, batch_size = self._prepare_views(batch)

        if masks is not None:
            outputs = self.model(views, masks)
        else:
            outputs = self.model(views)

        losses = self.criterion(outputs)
        loss = losses["total_loss"]
        return loss, losses, batch_size

    def prediction_step(self, batch: dict, mode: str) -> Tensor:
        loss, loss_components, batch_size = self.forward_step(batch)
        self.log(
            f"{mode}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        for name, value in loss_components.items():
            if name == "total_loss":
                continue
            self.log(
                f"{mode}/{name}",
                value,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self.prediction_step(batch, "training")

    def validation_step(self, batch, batch_idx):
        self.prediction_step(batch, "validation")

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update the EMA target network if the model supports it."""
        if self.momentum_scheduler is None:
            return
        if not hasattr(self.model, "update_target_network"):
            return

        current_step = self.trainer.global_step
        max_steps = self.trainer.max_steps if self.trainer.max_steps > 0 else (
            self.trainer.estimated_stepping_batches
        )
        momentum = self.momentum_scheduler(current_step, max_steps)
        self.model.update_target_network(momentum)
        self.log("training/ema_momentum", momentum, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.parameters())
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.scheduler_factory is None:
            return optimizer

        scheduler = self.scheduler_factory(optimizer)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": self.scheduler_interval,
            "frequency": 1,
        }

        if self.scheduler_monitor is not None:
            lr_scheduler_config["monitor"] = self.scheduler_monitor

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def load_pretraining_trainer(
    model: nn.Module,
    criterion: nn.Module,
    training_config: PretrainingConfig,
    momentum_scheduler: Optional[MomentumScheduler] = None,
    optimizer_factory: Optional[OptimizerFactory] = None,
    scheduler_factory: Optional[SchedulerFactory] = None,
    scheduler_interval: str = "step",
    scheduler_monitor: Optional[str] = None,
) -> PretrainingTrainer:
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")

    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return PretrainingTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            criterion=criterion,
            momentum_scheduler=momentum_scheduler,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_interval=scheduler_interval,
            scheduler_monitor=scheduler_monitor,
            weights_only=False,
        )

    return PretrainingTrainer(
        model=model,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        momentum_scheduler=momentum_scheduler,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        scheduler_interval=scheduler_interval,
        scheduler_monitor=scheduler_monitor,
    )