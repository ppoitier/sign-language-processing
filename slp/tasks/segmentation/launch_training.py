import os
import click

import torch
import numpy as np
torch.set_float32_matmul_precision('medium')

from slp.tasks.loggers import load_loggers
from slp.config.parser import parse_config
from slp.tasks.segmentation.config import SegmentationTaskConfig
from slp.tasks.segmentation.data import load_segmentation_datasets
from slp.tasks.segmentation.trainer import load_segmentation_trainer
from slp.tasks.training import run_training
from slp.tasks.testing import run_testing
from slp.utils.random import set_seed

from slp.nn.loading import load_model_architecture
from slp.losses.loading import load_criterion


def save_logits(lightning_module, config: SegmentationTaskConfig):
    print("Saving test logits...")
    experiment_config = config.experiment
    exp_name = f"{experiment_config.id}_{experiment_config.suffix}"
    logits_dir = f"{experiment_config.output_dir}/logits/{exp_name}"
    os.makedirs(logits_dir, exist_ok=True)
    logits = lightning_module.test_logits
    np.save(f"{logits_dir}/logits.npy", logits, allow_pickle=True)


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_segmentation_training(config_path):
    config: SegmentationTaskConfig = parse_config(config_path, SegmentationTaskConfig)
    print(config)
    set_seed(config.experiment.seed)

    datasets, dataloaders = load_segmentation_datasets(config.datasets)
    assert config.training is not None, "Missing training configuration."
    assert "training" in datasets, "Missing training dataset."
    assert "validation" in datasets, "Missing validation dataset."
    assert "testing" in datasets, "Missing testing dataset."

    model = load_model_architecture(config.model)
    criterion = load_criterion(datasets['training'], config.training)
    trainer = load_segmentation_trainer(model, criterion, config.training)

    lightning_module, best_checkpoint_path = run_training(
        training_dataloader=dataloaders['training'],
        validation_dataloader=dataloaders['validation'],
        lightning_module=trainer,
        experiment_config=config.experiment,
        training_config=config.training,
        loggers=load_loggers(config.experiment, prefix='train/'),
    )
    run_testing(
        checkpoint_path=best_checkpoint_path,
        testing_dataloader=dataloaders['testing'],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        loggers=load_loggers(config.experiment, 'test/'),
    )
    save_logits(lightning_module, config)


if __name__ == "__main__":
    launch_segmentation_training()
