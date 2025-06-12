import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def run_training(
    module,
    data_loaders,
    log_dir: str,
    checkpoints_dir: str,
    max_epochs: int,
    early_stopping_patience: int = 10,
    gradient_clipping: float = 0.0,
    debug: bool = False,
    show_progress_bar: bool = False,
):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        save_top_k=1,
        save_last=True,
        monitor="validation_loss",
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=True,
            log_weight_decay=True,
        ),
        EarlyStopping(
            monitor="validation_loss",
            patience=early_stopping_patience,
        ),
    ]
    tb_logger = TensorBoardLogger(name="tb", save_dir=log_dir)
    csv_logger = CSVLogger(name="csv", save_dir=log_dir)
    trainer = pl.Trainer(
        fast_dev_run=debug,
        gradient_clip_val=gradient_clipping,
        max_epochs=max_epochs,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        enable_progress_bar=show_progress_bar,
    )
    trainer.fit(
        module,
        train_dataloaders=data_loaders["training"],
        val_dataloaders=data_loaders["validation"],
    )
    return checkpoint_callback.best_model_path


def run_testing(
    module,
    data_loaders,
    log_dir: str,
    checkpoint_path: str | None = None,
    debug: bool = False,
    show_progress_bar: bool = False,
):
    tb_logger = TensorBoardLogger(name="tb", save_dir=log_dir)
    csv_logger = CSVLogger(name="csv", save_dir=log_dir)
    trainer = pl.Trainer(
        fast_dev_run=debug,
        logger=[tb_logger, csv_logger],
        enable_progress_bar=show_progress_bar,
    )
    trainer.test(
        module,
        ckpt_path=checkpoint_path,
        dataloaders=data_loaders["testing"],
    )
