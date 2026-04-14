import os

from lightning.pytorch.loggers import Logger, TensorBoardLogger, CSVLogger, MLFlowLogger

from slp.core.config.experiment import ExperimentConfig


def load_loggers(logs_dir: str, experiment_config: ExperimentConfig):
    os.makedirs(logs_dir, exist_ok=True)
    loggers: list[Logger] = [
        TensorBoardLogger(name="tb", save_dir=logs_dir),
        CSVLogger(name="csv", save_dir=logs_dir),
    ]
    if experiment_config.mlflow_uri is not None:
        loggers.append(
            MLFlowLogger(
                experiment_name=experiment_config.id,
                tracking_uri=experiment_config.mlflow_uri,
                # log_model=True,
            )
        )
    return loggers