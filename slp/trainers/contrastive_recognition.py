from torch import nn, optim

from .base import TrainerBase
from slp.metrics.segmentation.frame_based_group import FrameBasedMetrics
from slp.schedulers.cosine_annealing_with_linear_warmup import CosineAnnealingWithLinearWarmup


class ContrastiveRecognitionTrainer(TrainerBase):
    def __init__(
            self,
            backbone: nn.Module,
            projector: nn.Module,
            criterion: nn.Module,
            learning_rate: float,
            n_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        self.metrics_train = FrameBasedMetrics(prefix='train/', n_classes=n_classes)
        self.metrics_val = FrameBasedMetrics(prefix='val/', n_classes=n_classes)
        self.metrics_test = FrameBasedMetrics(prefix='test/', n_classes=n_classes)

        self.test_results = []
        self.save_hyperparameters(ignore=['backbone', 'projector', 'criterion', 'test_results'])

    def prediction_step(self, batch, mode):
        features, masks, labels = batch['poses'], batch['masks'], batch['label']
        targets = batch['label']
        batch_size = targets.shape[0]
        embeddings = self.backbone(features, masks)
        projections = self.projector(embeddings)
        loss = self.criterion(projections, labels)
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        return embeddings, projections, loss

    def training_step(self, batch, batch_idx):
        logits, projections, loss = self.prediction_step(batch, 'training')
        return loss

    def validation_step(self, batch, batch_index):
        self.prediction_step(batch, mode='validation')

    def test_step(self, batch, batch_index):
        ...

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWithLinearWarmup(optimizer, n_warmup_steps=20, max_steps=100, lr=self.learning_rate, start_lr=1e-8, end_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
