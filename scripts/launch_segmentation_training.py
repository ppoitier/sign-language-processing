from pprint import pprint

from slp.core.parser import parse_config
from slp.core.config.experiment import SegmentationTaskConfig

def launch_segmentation_training(config_path):
    config: SegmentationTaskConfig = parse_config(config_path, SegmentationTaskConfig)
    pprint(config)

    # set_seed(config.experiment.seed)

    # datasets, dataloaders = load_segmentation_datasets(config.datasets)
    # assert config.training is not None, "Missing training configuration."
    # assert "training" in datasets, "Missing training dataset."
    # assert "validation" in datasets, "Missing validation dataset."
    # assert "testing" in datasets, "Missing testing dataset."
    #
    # model = load_model_architecture(config.model)
    # criterion = load_criterion(datasets['training'], config.training)
    # trainer = load_segmentation_trainer(model, criterion, config.training)
    #
    # lightning_module, best_checkpoint_path = run_training(
    #     training_dataloader=dataloaders['training'],
    #     validation_dataloader=dataloaders['validation'],
    #     lightning_module=trainer,
    #     experiment_config=config.experiment,
    #     training_config=config.training,
    #     loggers=load_loggers(config.experiment, prefix='train/'),
    # )
    # run_testing(
    #     checkpoint_path=best_checkpoint_path,
    #     testing_dataloader=dataloaders['testing'],
    #     lightning_module=lightning_module,
    #     experiment_config=config.experiment,
    #     loggers=load_loggers(config.experiment, 'test/'),
    # )
    # save_logits(lightning_module, config)


if __name__ == "__main__":
    launch_segmentation_training("../config/1.actionness_with_weights.yaml")