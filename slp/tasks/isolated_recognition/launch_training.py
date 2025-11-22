import click

import torch
torch.set_float32_matmul_precision('high')

from slp.config.parser import parse_config
from slp.tasks.isolated_recognition.config import IsolatedRecognitionTaskConfig
from slp.tasks.isolated_recognition.data import load_isolated_recognition_datasets
from slp.tasks.isolated_recognition.trainer import load_isolated_recognition_trainer
from slp.tasks.training import run_training
from slp.tasks.testing import run_testing
from slp.tasks.loggers import load_loggers
from slp.utils.random import set_seed

from slp.nn.loading import load_model_architecture
from slp.losses.loading import load_criterion


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_isolated_recognition_training(config_path):
    config: IsolatedRecognitionTaskConfig = parse_config(config_path, IsolatedRecognitionTaskConfig)
    print(config)
    set_seed(config.experiment.seed)
    datasets, dataloaders = load_isolated_recognition_datasets(config.datasets)
    assert config.training is not None, "Missing training configuration."
    assert "training" in datasets, "Missing training dataset."
    assert "validation" in datasets, "Missing validation dataset."
    assert "testing" in datasets, "Missing testing dataset."

    model = load_model_architecture(config.model)
    criterion = load_criterion(datasets["training"], config.training)
    print("Model architecture:")
    print(model)

    lightning_module = load_isolated_recognition_trainer(model, criterion, config.training)
    lightning_module, best_checkpoint_path = run_training(
        training_dataloader=dataloaders["training"],
        validation_dataloader=dataloaders["validation"],
        lightning_module=lightning_module,
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
