import os

from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from slp.config.templates.experiment import ExperimentConfig


def run_testing(
    checkpoint_path: str,
    testing_dataloader: DataLoader,
    lightning_module: pl.LightningModule,
    experiment_config: ExperimentConfig,
):
    exp_name = f"{experiment_config.id}_{experiment_config.suffix}"
    logs_dir = f"{experiment_config.output_dir}/{exp_name}/logs"
    os.makedirs(logs_dir, exist_ok=True)
    if experiment_config.debug:
        checkpoint_path = None

    tb_logger = TensorBoardLogger(name="tb", save_dir=logs_dir)
    csv_logger = CSVLogger(name="csv", save_dir=logs_dir)
    lightning_trainer = pl.Trainer(
        fast_dev_run=experiment_config.debug,
        logger=[tb_logger, csv_logger],
        enable_progress_bar=experiment_config.show_progress_bar,
    )
    lightning_trainer.test(
        lightning_module,
        ckpt_path=checkpoint_path,
        dataloaders=testing_dataloader,
    )
