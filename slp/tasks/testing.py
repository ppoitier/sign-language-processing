from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import Logger

from slp.config.templates.experiment import ExperimentConfig


def run_testing(
    checkpoint_path: str | None,
    testing_dataloader: DataLoader,
    lightning_module: pl.LightningModule,
    experiment_config: ExperimentConfig,
    loggers: list[Logger],
):
    if experiment_config.debug:
        checkpoint_path = None
    lightning_trainer = pl.Trainer(
        fast_dev_run=experiment_config.debug,
        logger=loggers,
        enable_progress_bar=experiment_config.show_progress_bar,
    )
    lightning_trainer.test(
        lightning_module,
        ckpt_path=checkpoint_path,
        dataloaders=testing_dataloader,
    )
