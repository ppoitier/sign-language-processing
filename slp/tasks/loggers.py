import os

from lightning.pytorch.loggers import Logger, TensorBoardLogger, CSVLogger, MLFlowLogger

from slp.config.templates.experiment import ExperimentConfig


def load_loggers(experiment_config: ExperimentConfig, prefix: str = ''):
    exp_name = f"{experiment_config.id}_{experiment_config.suffix}"
    logs_dir = f"{experiment_config.output_dir}/logs/{exp_name}"
    os.makedirs(logs_dir, exist_ok=True)
    loggers: list[Logger] = [
        TensorBoardLogger(name="tb", save_dir=logs_dir),
        CSVLogger(name="csv", save_dir=logs_dir),
    ]
    if experiment_config.mlflow_uri is not None:
        loggers.append(
            MLFlowLogger(
                experiment_name=experiment_config.id,
                run_name=f"{prefix}{experiment_config.suffix}",
                tracking_uri=experiment_config.mlflow_uri,
                # log_model=True,
            )
        )
    return loggers
