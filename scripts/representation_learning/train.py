from pprint import pprint

import click

import torch

torch.set_float32_matmul_precision("medium")

from slp.core.parser import parse_config
from slp.core.config.experiment import RepresentationLearningTaskConfig
from slp.datasets.loading import load_isolated_datasets_and_loaders
from slp.nn.model_builder import build_hydra_model
from slp.nn.ssl.loading import infer_embedding_dim
from slp.utils.random import set_seed
from slp.utils.loggers import load_loggers
from slp.trainers.representation_learning import load_representation_learning_trainer
from slp.training import run_training
from slp.schedulers.loading import load_lr_scheduler_factory


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the YAML/JSON self-supervised pretraining configuration file.",
)
def launch_representation_learning(config_path):
    config: RepresentationLearningTaskConfig = parse_config(
        config_path, RepresentationLearningTaskConfig
    )
    pprint(config)

    selected_seed = set_seed(config.experiment.seed)
    print("Using seed: ", selected_seed)

    print("Loading datasets...")
    datasets, dataloaders = load_isolated_datasets_and_loaders(config.datasets)
    print(datasets.keys())

    print("Building model...")
    model = build_hydra_model(config.model)
    print(model)

    assert config.training is not None, "Missing training configuration."

    lr_scheduler_factory, lr_scheduler_monitor = None, None
    if config.training.lr_scheduler:
        print("Loading learning rate scheduler factory...")
        lr_scheduler_factory, lr_scheduler_monitor = load_lr_scheduler_factory(
            config.training.lr_scheduler
        )

    print("Loading representation learning trainer...")
    lightning_module = load_representation_learning_trainer(
        model=model,
        training_config=config.training,
        # Saves repeating the projector width in the method kwargs.
        embedding_dim=infer_embedding_dim(
            config.model, config.training.embedding_head
        ),
        scheduler_factory=lr_scheduler_factory,
        scheduler_monitor=lr_scheduler_monitor,
    )

    exp_config = config.experiment
    checkpoints_dir = f"{exp_config.output_dir}/checkpoints/{exp_config.id}/{exp_config.variant}/{selected_seed}"
    logs_dir = f"{exp_config.output_dir}/logs/{exp_config.id}/{exp_config.variant}/{selected_seed}"
    loggers = load_loggers(logs_dir, exp_config)

    lightning_module, best_checkpoint_path = run_training(
        training_dataloader=dataloaders["training"],
        validation_dataloader=dataloaders["validation"],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        training_config=config.training,
        loggers=loggers,
        checkpoints_dir=checkpoints_dir,
        monitor_loss="validation/loss",
    )

    # No test phase here: a pretrained encoder is evaluated by the downstream
    # task, by pointing an ISLR/segmentation config at this checkpoint.
    print("Pretrained checkpoint:", best_checkpoint_path)


if __name__ == "__main__":
    launch_representation_learning()
