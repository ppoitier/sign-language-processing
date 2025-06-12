import click
import random
import numpy as np
import torch

from slp.config.load_config import load_contrastive_recognition_task_config
from slp.config.loading.tasks import load_contrastive_recognition_task
from slp.trainers.utils import run_training, run_testing


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_contrastive_recognition_training(config_path):
    config = load_contrastive_recognition_task_config(config_path)

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_id, data_sets, data_loaders, module = load_contrastive_recognition_task(config)
    exp_name = f"{config.prefix}_{task_id}_{str(int(config.task_datetime.timestamp() * 1000))}"
    log_dir = f"{config.output_dir}/logs/{exp_name}"
    checkpoints_dir = f"{config.output_dir}/checkpoints/{exp_name}"
    print("Logs directory:", checkpoints_dir)
    print("Checkpoints directory:", log_dir)
    best_checkpoint_path = run_training(
        module,
        data_loaders,
        gradient_clipping=config.training.gradient_clipping,
        log_dir=log_dir,
        checkpoints_dir=checkpoints_dir,
        max_epochs=config.training.max_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        debug=config.training.debug,
        show_progress_bar=False,
    )
    # run_testing(
    #     module,
    #     data_loaders,
    #     log_dir=log_dir,
    #     checkpoint_path=best_checkpoint_path,
    #     debug=config.training.debug,
    # )


if __name__ == "__main__":
    launch_contrastive_recognition_training()
