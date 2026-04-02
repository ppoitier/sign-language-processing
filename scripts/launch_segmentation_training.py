from pprint import pprint

from slp.core.parser import parse_config
from slp.core.config.experiment import SegmentationTaskConfig
from slp.datasets.loading import load_continuous_datasets_and_loaders
from slp.nn.model_builder import build_hydra_model
from slp.nn.losses.loading import build_multi_layer_criterion
from slp.utils.random import set_seed
from slp.trainers.segmentation import load_segmentation_trainer
from slp.decoders.argmax import ArgmaxDecoder
from slp.training import run_training
from slp.testing import run_testing


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

    print("Building criterion...")
    criterion = build_multi_layer_criterion(config.training)
    print(criterion)

    print("Loading segmentation trainer...")
    lightning_module = load_segmentation_trainer(
        model=model,
        criterion=criterion,
        training_config=config.training,
        segment_decoder=ArgmaxDecoder(),
    )

    lightning_module, best_checkpoint_path = run_training(
        training_dataloader=dataloaders['training'],
        validation_dataloader=dataloaders['validation'],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        training_config=config.training,
        monitor_loss='validation/loss',
    )

    run_testing(
        checkpoint_path=best_checkpoint_path,
        testing_dataloader=dataloaders['testing'],
        lightning_module=lightning_module,
        experiment_config=config.experiment,
        # loggers=load_loggers(config.experiment, 'test/'),
    )


if __name__ == "__main__":
    launch_segmentation_training("../config/1.actionness.yaml")
