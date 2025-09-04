import torch

from slp.config.loading.backbone import load_backbone
from slp.config.loading.criterion import (
    load_contrastive_criterion,
    load_recognition_criterion,
)
from slp.config.loading.projector import load_projector
from slp.config.templates.task import SegmentationTaskConfig, ContrastiveRecognitionTask, RecognitionTaskConfig
from slp.trainers.contrastive_recognition import ContrastiveRecognitionTrainer
from slp.trainers.recognition import RecognitionTrainer


def load_recognition_lightning_module(
    config: RecognitionTaskConfig, training_dataset=None
):
    criterion_weights = None
    if config.training.criterion.use_weights:
        assert (
            training_dataset is not None
        ), "Training dataset is required to use class weights."
        criterion_weights = torch.from_numpy(
            training_dataset.get_label_weights()
        ).float()
        print("Use criterion weights:", criterion_weights.tolist())
    backbone = load_backbone(config.backbone)
    criterion = load_recognition_criterion(
        config.training.criterion, criterion_weights=criterion_weights
    )
    checkpoint_path = config.backbone.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return RecognitionTrainer.load_from_checkpoint(
            checkpoint_path, backbone=backbone, criterion=criterion
        )
    return RecognitionTrainer(
        backbone=backbone,
        criterion=criterion,
        learning_rate=config.training.learning_rate,
        n_classes=config.training.criterion.n_classes,
    )


def load_contrastive_recognition_lightning_module(
    config: ContrastiveRecognitionTask, training_dataset=None
):
    criterion_weights = None
    if config.training.criterion.use_weights:
        assert (
            training_dataset is not None
        ), "Training dataset is required to use class weights."
        criterion_weights = torch.from_numpy(
            training_dataset.get_label_weights(config.target_codec.name)
        ).float()
        print("Use criterion weights:", criterion_weights.tolist())
    backbone = load_backbone(config.backbone)
    projector = load_projector(config.projector)
    criterion = load_contrastive_criterion(
        config.training.criterion, criterion_weights=criterion_weights
    )
    checkpoint_path = config.backbone.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return ContrastiveRecognitionTrainer.load_from_checkpoint(
            checkpoint_path, backbone=backbone, projector=projector, criterion=criterion
        )
    return ContrastiveRecognitionTrainer(
        backbone=backbone,
        projector=projector,
        criterion=criterion,
        learning_rate=config.training.learning_rate,
        n_classes=config.training.criterion.n_classes,
    )
