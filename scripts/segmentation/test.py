import click
import os

from slp.config.load_config import load_segmentation_task_config
from slp.config.loading.tasks import load_segmentation_task
from slp.trainers.utils import run_training, run_testing


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_segmentation_testing(config_path):
    config = load_segmentation_task_config(config_path)
    task_id, data_sets, data_loaders, module = load_segmentation_task(config)
    exp_name = f"{task_id}_{str(int(config.task_datetime.timestamp() * 1000))}"
    log_dir = f"{config.output_dir}/logs/{exp_name}"
    if config.backbone.checkpoint_path is None:
        raise ValueError("Checkpoint path is required for testing.")
    if not os.path.isfile(config.backbone.checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found.")
    print("Logs directory:", log_dir)
    run_testing(
        module,
        data_loaders,
        log_dir=log_dir,
        checkpoint_path=config.backbone.checkpoint_path,
        debug=config.training.debug,
    )


if __name__ == "__main__":
    launch_segmentation_testing()
