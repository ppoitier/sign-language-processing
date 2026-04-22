from pprint import pprint

import click
import torch
torch.set_float32_matmul_precision("medium")

from slp.core.parser import parse_config
from slp.core.config.experiment import SegmentationTaskConfig
from slp.datasets.loading import load_continuous_datasets_and_loaders
from slp.nn.model_builder import build_hydra_model
from slp.nn.losses.loading import build_multi_layer_criterion
from slp.utils.random import set_seed
from slp.utils.loggers import load_loggers
from slp.trainers.segmentation import load_segmentation_trainer
from slp.decoders.loading import load_segment_decoder
from slp.training import run_training
from slp.testing import run_testing
from slp.utils.logits import save_logits
from slp.schedulers.loading import load_lr_scheduler_factory


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the YAML/JSON segmentation training configuration file.",
)
def launch_segmentation_training(config_path):
    config: SegmentationTaskConfig = parse_config(config_path, SegmentationTaskConfig)
    pprint(config)

    selected_seed = set_seed(config.experiment.seed)
    print("Using seed: ", selected_seed)

    print("Loading datasets...")
    datasets, dataloaders = load_continuous_datasets_and_loaders(config.datasets)
    print(datasets.keys())

    print("Building model...")
    model = build_hydra_model(config.model)
    print(model)

    print("Compiling model...")
    model = torch.compile(model)

    assert config.training is not None, "Missing training configuration."

    print("Building criterion...")
    criterion = build_multi_layer_criterion(config.training, datasets["training"])
    print(criterion)

    print("Loading segment decoder...")
    segment_decoder = load_segment_decoder(config.training.segment_decoder)

    print("Loading learning rate scheduler factory...")
    if config.training.lr_scheduler is not None:
        lr_scheduler_factory, lr_scheduler_monitor = load_lr_scheduler_factory(config.training.lr_scheduler)
    else:
        lr_scheduler_factory, lr_scheduler_monitor = None, None

    print("Loading segmentation trainer...")
    lightning_module = load_segmentation_trainer(
        model=model,
        criterion=criterion,
        training_config=config.training,
        segment_decoder=segment_decoder,
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

    run_testing(
        checkpoint_path=best_checkpoint_path,
        testing_dataloader=dataloaders["testing"],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        loggers=loggers,
    )

    run_testing(
        checkpoint_path=best_checkpoint_path,
        testing_dataloader=dataloaders["training"],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        loggers=None,
    )

    logits_dir = f"{exp_config.output_dir}/logits/{exp_config.id}/{exp_config.variant}/{selected_seed}"
    save_logits(lightning_module.test_logits, logits_dir)


if __name__ == "__main__":
    launch_segmentation_training()
