from typing import Optional

from torch import Tensor, optim
import lightning as pl

from slp.schedulers.types import OptimizerFactory, SchedulerFactory


class TrainerBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_metrics(
        self,
        metrics: dict[str, any],
        batch_size: int,
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        for name, value in metrics.items():
            if isinstance(value, Tensor) and value.numel() > 1:
                assert len(value.shape) == 1
                for idx, v in enumerate(value.tolist()):
                    self.log(
                        f"{name}/{idx}",
                        v,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        batch_size=batch_size,
                    )
            else:
                self.log(
                    name,
                    value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )

    def setup_optimization(
        self,
        learning_rate: float,
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_interval: str = "epoch",
        scheduler_monitor: Optional[str] = None,
    ) -> None:
        """Store the optimizer/scheduler plumbing read by ``configure_optimizers``.

        Call this from a subclass ``__init__``. Kept out of ``__init__`` so that
        subclasses stay free to declare their own explicit signature, which
        ``save_hyperparameters`` relies on.
        """
        self.learning_rate = learning_rate
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.scheduler_interval = scheduler_interval
        self.scheduler_monitor = scheduler_monitor

    def optimized_parameters(self):
        """Parameters handed to the optimizer.

        Override to exclude frozen sub-modules (e.g. an EMA target encoder).
        """
        return self.parameters()

    def configure_optimizers(self):
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.optimized_parameters())
        else:
            optimizer = optim.AdamW(self.optimized_parameters(), lr=self.learning_rate)

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
