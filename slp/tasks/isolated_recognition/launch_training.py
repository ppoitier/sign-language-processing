import click

import torch
torch.set_float32_matmul_precision('medium')

from slp.config.parser import parse_config
from slp.tasks.isolated_recognition.config import IsolatedRecognitionTaskConfig
from slp.tasks.isolated_recognition.data import load_isolated_recognition_datasets
from slp.tasks.isolated_recognition.trainer import load_isolated_recognition_trainer
from slp.tasks.training import run_training
from slp.tasks.testing import run_testing
from slp.tasks.loggers import load_loggers
from slp.utils.random import set_seed


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_isolated_recognition_training(config_path):
    config: IsolatedRecognitionTaskConfig = parse_config(config_path, IsolatedRecognitionTaskConfig)
    print(config)
    set_seed(config.experiment.seed)
    datasets, dataloaders = load_isolated_recognition_datasets(config)
    assert config.training is not None, "Missing training configuration."
    assert "training" in datasets, "Missing training dataset."
    assert "validation" in datasets, "Missing validation dataset."
    assert "testing" in datasets, "Missing testing dataset."

    trainer = load_isolated_recognition_trainer(
        datasets["training"], config.model, config.training
    )
    lightning_module, best_checkpoint_path = run_training(
        training_dataloader=dataloaders["training"],
        validation_dataloader=dataloaders["validation"],
        lightning_module=trainer,
        experiment_config=config.experiment,
        training_config=config.training,
        loggers=load_loggers(config.experiment, prefix="train/"),
    )
    run_testing(
        checkpoint_path=best_checkpoint_path,
        testing_dataloader=dataloaders["testing"],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        loggers=load_loggers(config.experiment, "test/"),
    )


if __name__ == "__main__":
    launch_isolated_recognition_training()
