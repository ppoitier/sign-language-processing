from torch import nn, optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ReduceLROnPlateau

from .base import TrainerBase
from slp.metrics.segmentation.frame_based_group import FrameBasedMetrics
from slp.schedulers.warmup_on_plateau import WarmupReduceLROnPlateau


class RecognitionTrainer(TrainerBase):
    def __init__(
            self,
            backbone: nn.Module,
            criterion: nn.Module,
            learning_rate: float,
            n_classes: int,
            n_warmup_epochs: int = 5,
            plateau_factor: float = 0.1,
            plateau_patience: int = 3,
    ):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.n_warmup_epochs = n_warmup_epochs
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience

        self.metrics_train = FrameBasedMetrics(prefix='train/', n_classes=n_classes)
        self.metrics_val = FrameBasedMetrics(prefix='val/', n_classes=n_classes)
        self.metrics_test = FrameBasedMetrics(prefix='test/', n_classes=n_classes)

        self.test_results = []
        self.save_hyperparameters(ignore=['backbone', 'criterion', 'test_results'])

    def prediction_step(self, batch, mode):
        features, labels = batch['poses'], batch['label']
        targets = batch['label']
        batch_size = targets.shape[0]
        logits = self.backbone(features)
        loss = self.criterion(logits, labels)
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        probs = logits.softmax(dim=-1)
        if mode == 'training':
            metrics = self.metrics_train(probs, targets)
        elif mode == 'validation':
            metrics = self.metrics_val(probs, targets)
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
        ...

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=self.n_warmup_epochs,
            patience=self.plateau_patience,
            factor=self.plateau_factor,
            mode='min',
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
