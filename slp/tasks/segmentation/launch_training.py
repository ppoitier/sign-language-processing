import click
from pprint import pprint

import torch
torch.set_float32_matmul_precision('medium')

from slp.config.parser import parse_config
from slp.tasks.segmentation.config import SegmentationTaskConfig
from slp.tasks.segmentation.data import load_segmentation_datasets
from slp.tasks.segmentation.trainer import load_segmentation_trainer
from slp.tasks.training import run_training
from slp.tasks.testing import run_testing
from slp.utils.random import set_seed


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_segmentation_training(config_path):
    config: SegmentationTaskConfig = parse_config(config_path, SegmentationTaskConfig)
    pprint(config)
    set_seed(config.experiment.seed)
    datasets, dataloaders = load_segmentation_datasets(config)
    assert config.training is not None, "Missing training configuration."
    assert "training" in datasets, "Missing training dataset."
    assert "validation" in datasets, "Missing validation dataset."
    trainer = load_segmentation_trainer(datasets['training'], config.model, config.training)
    lightning_module, best_checkpoint_path = run_training(
        training_dataloader=dataloaders['training'],
        validation_dataloader=dataloaders['validation'],
        lightning_module=trainer,
        experiment_config=config.experiment,
        training_config=config.training,
    )
    run_testing(
        checkpoint_path=best_checkpoint_path,
        testing_dataloader=dataloaders['testing'],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
    )


if __name__ == "__main__":
    launch_segmentation_training()
