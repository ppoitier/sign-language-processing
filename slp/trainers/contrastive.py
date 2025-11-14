from torch import nn, optim

from slp.trainers.base import TrainerBase
from slp.schedulers.cosine_annealing_with_linear_warmup import create_warmup_plateau_cosine_scheduler
from slp.config.templates.training import TrainingConfig
from slp.utils.model import count_parameters


class ContrastiveIsolatedRecognitionTrainer(TrainerBase):
    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        criterion: nn.Module,
        lr: float,
        n_epochs: int,
        n_warmup_epochs: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.criterion = criterion

        self.lr = lr
        self.n_epochs = n_epochs
        self.n_warmup_epochs = n_warmup_epochs

        self.test_results = []
        self.save_hyperparameters(ignore=["backbone", "projector", "criterion", "test_results"])

    def pred_step(self, batch, mode: str):
        features, masks, targets = batch["poses"].float(), batch["masks"].bool(), batch["label_id"].long()
        batch_size = features.size(0)
        embeddings = self.backbone(features, masks)
        projections = self.projector(embeddings)
        contrastive_loss = self.criterion(projections, targets)
        self.log(f"{mode}/contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        return embeddings, contrastive_loss

    def training_step(self, batch, batch_idx):
        _, loss = self.pred_step(batch, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        self.pred_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        instance_ids, lengths, targets = batch["id"], batch["length"], batch["label_id"]
        embeddings, loss = self.pred_step(batch, "training")
        # todo

    def on_test_epoch_end(self):
        ... # todo
        # all_embeddings = torch.cat(self.test_results['embeddings'])
        # all_labels = torch.cat(self.test_results['labels'])
        # all_ids = sum(self.test_results['ids'], start=tuple())
        # self.test_results = {'embeddings': all_embeddings, 'labels': all_labels, 'ids': all_ids}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = create_warmup_plateau_cosine_scheduler(
            optimizer=optimizer,
            n_warmup_steps=self.n_warmup_epochs,
            n_plateau_steps=0,
            max_steps=self.n_epochs,
            lr=self.lr,
            start_lr=1e-6,
            end_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def load_contrastive_isolated_recognition_trainer(
    backbone: nn.Module,
    projector: nn.Module,
    criterion: nn.Module,
    training_config: TrainingConfig,
):
    print(f"Total number of parameters in the backbone: {count_parameters(backbone):,}")
    print(f"Total number of parameters in the projector: {count_parameters(projector):,}")
    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return ContrastiveIsolatedRecognitionTrainer.load_from_checkpoint(checkpoint_path, backbone=backbone, projector=projector, criterion=criterion)
    if training_config.n_warmup_epochs is None:
        raise ValueError("You need to provide a number of warmup epochs for contrastive models.")
    return ContrastiveIsolatedRecognitionTrainer(
        backbone=backbone,
        projector=projector,
        criterion=criterion,
        lr=training_config.learning_rate,
        n_epochs=training_config.max_epochs,
        n_warmup_epochs=training_config.n_warmup_epochs,
    )
