import os

import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader

from slp.core.config.experiment import ExperimentConfig
from slp.core.config.training import TrainingConfig


def run_training(
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    lightning_module: pl.LightningModule,
    experiment_config: ExperimentConfig,
    training_config: TrainingConfig,
    loggers: list[Logger],
    checkpoints_dir: str,
    monitor_loss: str = 'validation/loss',
) -> tuple[pl.LightningModule, str]:

    # identifier = f"{experiment_config.task}/{experiment_config.variant}/{experiment_config.id}"
    # checkpoints_dir = f"{experiment_config.output_dir}/checkpoints/{identifier}"
    # os.makedirs(checkpoints_dir, exist_ok=True)
    # log_dir = f"{experiment_config.output_dir}/logs/{identifier}"

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
            log_momentum=False,
            log_weight_decay=False,
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
        # logger=[TensorBoardLogger(save_dir=log_dir, name='tb')],
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
