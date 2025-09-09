import torch
from torch import nn, optim

from slp.config.templates.model import ModelConfig
from slp.config.templates.training import TrainingConfig
from slp.data.datasets.densely_annotated import DenselyAnnotatedSLDataset
from slp.losses.loading import load_multihead_criterion
from slp.metrics.segmentation.frame_based_group import FrameBasedMetrics
from slp.metrics.segmentation.segment_based_group import SegmentBasedMetrics
from slp.tasks.segmentation.model import load_model
from slp.trainers.base import TrainerBase
from slp.utils.model import count_parameters
from slp.codecs.annotations.base import AnnotationCodec
from slp.codecs.annotations.offsets import OffsetsCodec
from slp.codecs.annotations.loading import load_annotation_codec


class SegmentationTrainer(TrainerBase):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        n_classes: int,
        is_output_multilayer: bool,
        use_offsets: bool,
        heads_to_targets: dict[str, str],
        segment_decoder: AnnotationCodec,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.is_output_multilayer = is_output_multilayer
        self.use_offsets = use_offsets
        self.heads_to_targets = heads_to_targets
        self.segment_decoder = segment_decoder

        self.frame_metrics_train = FrameBasedMetrics(prefix="training/frames/", n_classes=n_classes)
        self.frame_metrics_val = FrameBasedMetrics(prefix="validation/frames/", n_classes=n_classes)
        self.frame_metrics_test = FrameBasedMetrics(prefix="testing/frames/", n_classes=n_classes)

        self.segment_metrics_test = SegmentBasedMetrics(prefix="testing/segments/")

        self.test_metrics = {}
        self.test_logits = {}
        self.save_hyperparameters(ignore=["model", "criterion", "test_results"])

    def prediction_step(self, batch, mode):
        features, masks, targets = batch["poses"], batch["masks"], batch["targets"]
        batch_size = features.size(0)
        # features of shape (N, C_in, T)
        # mask of shape (N, 1, T)
        # classification target of shape (N, T)
        # offsets target of shape (N, 2, T)

        logits = self.model(features, masks)
        loss = self.criterion(
            logits,
            {
                head_name: targets[target_name]
                for head_name, target_name in self.heads_to_targets.items()
            },
        )

        self.log(
            f"{mode}/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size
        )
        cls_logits = logits["classification"]
        cls_logits = cls_logits[-1] if self.is_output_multilayer else cls_logits
        per_frame_probs = cls_logits.softmax(dim=1)
        if mode == "training":
            metrics = self.frame_metrics_train(per_frame_probs, targets["frame_labels"])
        elif mode == "validation":
            metrics = self.frame_metrics_val(per_frame_probs, targets["frame_labels"])
        elif mode == "testing":
            metrics = self.frame_metrics_test(per_frame_probs, targets["frame_labels"])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.log_metrics(metrics, batch_size=batch_size)
        return logits, loss, metrics

    def training_step(self, batch, batch_idx):
        _, loss, _ = self.prediction_step(batch, "training")
        return loss

    def validation_step(self, batch, batch_index):
        self.prediction_step(batch, mode="validation")

    def test_step(self, batch, batch_index):
        instance_ids, starts, ends, lengths, gt_segments = (
            batch["id"],
            batch["start"],
            batch["end"],
            batch["length"],
            batch["targets"]["segments"],
        )
        logits, loss, frame_metrics = self.prediction_step(batch, "testing")
        if self.is_output_multilayer:
            logits = {k: v[-1] for k, v in logits.items()}

        batch_size = len(instance_ids)
        for idx in range(batch_size):
            instance_id = instance_ids[idx]
            self.test_logits[f"{instance_id}_{starts[idx]}_{ends[idx]}"] = {
                head_name: head_logits[idx]
                .detach()
                .cpu()
                .numpy()[..., : lengths[idx]]
                .astype("float16")
                for head_name, head_logits in logits.items()
            }
            # results.append(
            #     {
            #         "id": instance_ids[idx],
            #         "length": lengths[idx].item(),
            #         "segments": gt_segments[idx].detach().cpu().numpy().astype("int32"),
            #         "logits": {
            #             head_name: head_logits[idx]
            #             .detach()
            #             .cpu()
            #             .numpy()[..., : lengths[idx]]
            #             .astype("float16")
            #             for head_name, head_logits in logits.items()
            #         },
            #     }
            # )
        # self.test_results += results

        pred_segments = self.segment_decoder.decode_batch(logits, self.n_classes, batch_size)
        segment_metrics = self.segment_metrics_test(pred_segments, gt_segments)
        self.log_metrics(segment_metrics, batch_size=batch_size)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)


def load_segmentation_trainer(
    training_dataset: DenselyAnnotatedSLDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_weights = {}
    for name, config in training_config.criterion.items():
        if config.use_weights:
            weights = training_dataset.get_label_weights()
            print(f"Use weights for target:", weights)
            criterion_weights[name] = torch.from_numpy(weights).float().to(device)
    criterion = load_multihead_criterion(training_config.criterion, criterion_weights)
    model = load_model(model_config)
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")
    checkpoint_path = model_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return SegmentationTrainer.load_from_checkpoint(
            checkpoint_path, model=model, criterion=criterion
        )
    cls_head = model_config.heads.get("classification")
    if cls_head is None:
        raise ValueError("A segmentation model must have a 'classification' head.")
    return SegmentationTrainer(
        model=model,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        n_classes=cls_head.out_channels,
        is_output_multilayer=training_config.is_output_multilayer,
        use_offsets=training_config.use_offsets,
        heads_to_targets=training_config.heads_to_targets,
        segment_decoder=load_annotation_codec(training_config.segment_decoder)
    )
