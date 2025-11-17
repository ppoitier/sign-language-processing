import os

import lightning
import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import Logger, TensorBoardLogger, CSVLogger, MLFlowLogger
from lightning.pytorch.plugins import BitsandbytesPrecision
from torch.utils.data import DataLoader

from slp.config.templates.experiment import ExperimentConfig
from slp.config.templates.training import TrainingConfig


def run_training(
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    lightning_module: pl.LightningModule,
    experiment_config: ExperimentConfig,
    training_config: TrainingConfig,
    loggers: list[Logger],
    monitor_loss: str = 'validation/loss',
) -> tuple[lightning.LightningModule, str]:
    exp_name = f"{experiment_config.id}_{experiment_config.suffix}"
    checkpoints_dir = f"{experiment_config.output_dir}/checkpoints/{exp_name}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        save_top_k=1,
        save_last=True,
        monitor=monitor_loss,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=True,
            log_weight_decay=True,
        ),
        EarlyStopping(
            monitor=monitor_loss,
            patience=training_config.early_stopping_patience,
        ),
    ]
    lightning_trainer = pl.Trainer(
        fast_dev_run=experiment_config.debug,
        gradient_clip_val=training_config.gradient_clipping,
        max_epochs=training_config.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=experiment_config.show_progress_bar,
        overfit_batches=1 if training_config.overfit_one_batch else 0,
        # plugins=BitsandbytesPrecision('nf4-dq'),
    )
    lightning_trainer.fit(
        lightning_module,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )
    return lightning_module, checkpoint_callback.best_model_path
