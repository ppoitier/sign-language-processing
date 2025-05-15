import torch

from slp.config.templates.task import SegmentationTaskConfig
from slp.config.loading.codecs import load_segments_codec
from slp.config.loading.backbone import load_segmentation_backbone
from slp.config.loading.criterion import load_segmentation_criterion
from slp.trainers.segmentation import SegmentationTrainer


def load_segmentation_lightning_module(config: SegmentationTaskConfig, training_dataset=None):
    target_codec_name = config.target_codec.name
    target_codec = load_segments_codec(config.target_codec)
    backbone = load_segmentation_backbone(config.backbone)
    criterion_weights = None
    if config.training.criterion.use_weights:
        assert training_dataset is not None, "Training dataset is required to use class weights."
        criterion_weights = torch.from_numpy(training_dataset.get_label_weights(config.target_codec.name)).float()
        print("Use criterion weights:", criterion_weights.tolist())
    criterion = load_segmentation_criterion(config.training.criterion, criterion_weights=criterion_weights)
    checkpoint_path = config.backbone.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return SegmentationTrainer.load_from_checkpoint(checkpoint_path, backbone=backbone, criterion=criterion)
    return SegmentationTrainer(
        codec=(target_codec_name, target_codec),
        backbone=backbone,
        criterion=criterion,
        learning_rate=config.training.learning_rate,
        n_classes=config.training.criterion.n_classes,
        multi_layer_output=config.training.multi_layer_output,
        use_offsets=config.target_codec.use_offsets,
    )
