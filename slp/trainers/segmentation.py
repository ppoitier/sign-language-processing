import torch
from torch import nn, optim

from .base import TrainerBase
from slp.codecs.segmentation.base import SegmentationCodec
from slp.metrics.segmentation.segment_based_group import SegmentBasedMetrics
from slp.metrics.segmentation.frame_based_group import FrameBasedMetrics


class SegmentationTrainer(TrainerBase):
    def __init__(
            self,
            codec: tuple[str, SegmentationCodec],
            backbone: nn.Module,
            criterion: nn.Module,
            learning_rate: float,
            n_classes: int,
            multi_layer_output,
            use_offsets,
    ):
        super().__init__()
        self.codec_name, self.codec = codec
        self.backbone = backbone
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.multi_layer_output = multi_layer_output
        self.use_offsets = use_offsets

        self.frame_metrics_train = FrameBasedMetrics(prefix='train/', n_classes=n_classes)
        self.frame_metrics_val = FrameBasedMetrics(prefix='val/', n_classes=n_classes)
        self.frame_metrics_test = FrameBasedMetrics(prefix='test/', n_classes=n_classes)

        self.segment_metrics_test = SegmentBasedMetrics(prefix='test/')

        self.test_results = []
        self.save_hyperparameters(ignore=['backbone', 'criterion', 'test_results'])

    def prediction_step(self, batch, mode):
        features, masks = batch['poses'], batch['masks']
        targets = batch[self.codec_name]['frames']
        batch_size = targets.shape[0]
        logits = self.backbone(features, masks)

        loss = self.criterion(logits, targets)
        if self.use_offsets:
            loss, cls_loss, reg_loss = loss
            self.log(f'{mode}_cls_loss', cls_loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log(f'{mode}_reg_loss', reg_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)

        if self.multi_layer_output:
            logits = logits[-1]

        per_frame_probs = self.codec.decode_logits_to_frame_probabilities(logits, self.n_classes)
        if self.use_offsets:
            cls_targets = targets[:, :, 0]
        else:
            cls_targets = targets

        # Permute dimensions to correspond to the torchmetrics specifications
        per_frame_probs = per_frame_probs.transpose(-1, -2).contiguous()
        if mode == 'training':
            metrics = self.frame_metrics_train(per_frame_probs, cls_targets)
        elif mode == 'validation':
            metrics = self.frame_metrics_val(per_frame_probs, cls_targets)
        elif mode == 'testing':
            metrics = self.frame_metrics_test(per_frame_probs, cls_targets)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.log_metrics(metrics, batch_size=batch_size)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.prediction_step(batch, 'training')
        return loss

    def validation_step(self, batch, batch_index):
        self.prediction_step(batch, mode='validation')

    def test_step(self, batch, batch_index):
        instance_ids, lengths, gt_segments = batch['id'], batch['lengths'], batch[self.codec_name]['segments']
        logits, _ = self.prediction_step(batch, mode="testing")
        batch_size = len(gt_segments)
        self.test_results += [
            {
                "id": instance_id,
                "logits": instance_logits.detach().cpu()[:instance_length],
                "segments": instance_segments.detach().cpu(),
            }
            for instance_id, instance_logits, instance_length, instance_segments in zip(
                instance_ids, logits, lengths, gt_segments
            )
        ]
        segment_preds = self.codec.batch_decode_logits_to_segments(logits, self.n_classes)
        segment_metrics = self.segment_metrics_test(segment_preds, gt_segments)
        self.log_metrics(segment_metrics, batch_size=batch_size)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
