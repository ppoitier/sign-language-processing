import click

import torch
from torch import nn
torch.set_float32_matmul_precision('medium')

from pytorch_metric_learning.losses import SupConLoss

from slp.config.parser import parse_config
from slp.tasks.isolated_recognition.config import ContrastiveIsolatedRecognitionTaskConfig
from slp.tasks.isolated_recognition.data import load_isolated_recognition_datasets
from slp.trainers.contrastive import load_contrastive_isolated_recognition_trainer, ContrastiveIsolatedRecognitionTrainer
from slp.trainers.linear_evaluation import load_linear_evaluation_trainer
from slp.tasks.training import run_training
from slp.tasks.testing import run_testing
from slp.tasks.loggers import load_loggers
from slp.utils.random import set_seed

from slp.nn.loading import load_backbone, load_projector


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_contrastive_isolated_recognition_training(config_path):
    config: ContrastiveIsolatedRecognitionTaskConfig = parse_config(config_path, ContrastiveIsolatedRecognitionTaskConfig)
    print(config)
    set_seed(config.experiment.seed)
    datasets, dataloaders = load_isolated_recognition_datasets(config.datasets)
    assert "training" in datasets, "Missing training dataset."
    assert "validation" in datasets, "Missing validation dataset."
    assert "testing" in datasets, "Missing testing dataset."

    print("Pre-training using contrastive model...")
    backbone = load_backbone(config.model.backbone)
    projector = load_projector(config.model.projector)
    contrastive_criterion = SupConLoss()
    lightning_contrastive_module = load_contrastive_isolated_recognition_trainer(
        backbone=backbone,
        projector=projector,
        criterion=contrastive_criterion,
        training_config=config.contrastive_training,
    )

    if not config.skip_contrastive_training:
        lightning_contrastive_module, best_contrastive_checkpoint_path = run_training(
            training_dataloader=dataloaders["training"],
            validation_dataloader=dataloaders["validation"],
            lightning_module=lightning_contrastive_module,
            experiment_config=config.experiment,
            training_config=config.contrastive_training,
            loggers=load_loggers(config.experiment, prefix="train/"),
            monitor_loss='validation/contrastive_loss',
        )
        checkpoint_to_evaluate = best_contrastive_checkpoint_path
    else:
        checkpoint_to_evaluate = config.contrastive_training.checkpoint_path
    if checkpoint_to_evaluate is None:
        raise ValueError("You must train a contrastive module or specify a checkpoint path for the linear evaluation.")

    print(f"Loading best backbone weights from: {checkpoint_to_evaluate}")
    backbone.load_state_dict({
        k.replace('backbone.', '', 1): v
        for k, v in torch.load(
            checkpoint_to_evaluate,
            map_location='cpu',
            weights_only=True
        )['state_dict'].items()
        if k.startswith('backbone.')
    })

    print("Linear evaluation...")
    cls_head = nn.Linear(in_features=128, out_features=500)
    cls_criterion = nn.CrossEntropyLoss()
    lightning_evaluation_module = load_linear_evaluation_trainer(
        backbone=backbone,
        cls_head=cls_head,
        criterion=cls_criterion,
        training_config=config.linear_evaluation_training,
    )
    lightning_evaluation_module, best_linear_evaluation_checkpoint_path = run_training(
        training_dataloader=dataloaders["training"],
        validation_dataloader=dataloaders["validation"],
        lightning_module=lightning_evaluation_module,
        experiment_config=config.experiment,
        training_config=config.linear_evaluation_training,
        loggers=load_loggers(config.experiment, prefix="train/"),
        monitor_loss="validation/cls_loss",
    )

    run_testing(
        checkpoint_path=best_linear_evaluation_checkpoint_path,
        testing_dataloader=dataloaders["testing"],
        lightning_module=lightning_evaluation_module,
        experiment_config=config.experiment,
        loggers=load_loggers(config.experiment, "test/"),
    )


if __name__ == "__main__":
    launch_contrastive_isolated_recognition_training()
