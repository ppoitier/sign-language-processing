from torch import nn

from slp.core.config.training import SegmentationTrainingConfig
from slp.metrics.segmentation.frame_based_group import FrameBasedMetrics
from slp.metrics.segmentation.segment_based_group import SegmentBasedMetrics
from slp.decoders.base import SegmentDecoder
from slp.utils.model import count_parameters

from slp.trainers.generic import GenericTrainer


class SegmentationTrainer(GenericTrainer):
    """Trainer for temporal segmentation tasks.

    Adds frame-level metrics (train/val/test), segment-level metrics (test),
    and test-time segment decoding on top of GenericTrainer.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        heads_to_targets: dict[str, str],
        is_output_multilayer: bool,
        n_classes: int,
        classification_head: str = "classification",
        frame_labels_target: str = "temporal-segmentation",
        segments_target: str = "segments",
        segment_decoder: SegmentDecoder | None = None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            learning_rate=learning_rate,
            heads_to_targets=heads_to_targets,
            is_output_multilayer=is_output_multilayer,
        )
        self.n_classes = n_classes
        self.classification_head = classification_head
        self.frame_labels_target = frame_labels_target
        self.segments_target = segments_target
        self.segment_decoder = segment_decoder

        self.frame_metrics = nn.ModuleDict({
            "training_metrics": FrameBasedMetrics(prefix="training/frames/", n_classes=n_classes),
            "validation_metrics": FrameBasedMetrics(prefix="validation/frames/", n_classes=n_classes),
            "testing_metrics": FrameBasedMetrics(prefix="testing/frames/", n_classes=n_classes),
        })
        self.segment_metrics_test = SegmentBasedMetrics(prefix="testing/segments/")

        self.save_hyperparameters(ignore=["model", "criterion", "segment_decoder"])

    def compute_metrics(self, logits: dict, batch: dict, mode: str) -> dict:
        cls_logits = logits[self.classification_head]
        per_frame_probs = cls_logits.softmax(dim=1)
        frame_targets = batch["targets"][self.frame_labels_target]
        return self.frame_metrics[f'{mode}_metrics'](per_frame_probs, frame_targets)

    def on_test_batch(self, logits: dict, batch: dict, batch_size: int) -> None:
        if self.segment_decoder is None:
            return
        pred_segments = self.segment_decoder.decode_batch(
            logits, self.n_classes, batch_size
        )
        gt_segments = batch["targets"][self.segments_target]
        segment_metrics = self.segment_metrics_test(pred_segments, gt_segments)
        self.log_metrics(segment_metrics, batch_size=batch_size)


def load_segmentation_trainer(
    model: nn.Module,
    criterion: nn.Module,
    training_config: SegmentationTrainingConfig,
    segment_decoder: SegmentDecoder,
) -> SegmentationTrainer:
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")

    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return SegmentationTrainer.load_from_checkpoint(
            checkpoint_path, model=model, criterion=criterion, segment_decoder=segment_decoder, weights_only=False,
        )

    n_classes = training_config.n_classes
    if n_classes is None:
        raise ValueError(
            "The number of classes (n_classes) must be provided in the training configuration."
        )

    return SegmentationTrainer(
        model=model,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        heads_to_targets=training_config.heads_to_targets,
        is_output_multilayer=training_config.is_output_multilayer,
        n_classes=n_classes,
        segment_decoder=segment_decoder,
    )