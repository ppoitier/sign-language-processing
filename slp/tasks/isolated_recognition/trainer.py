import torch
from torch import nn, optim

from slp.config.templates.model import ModelConfig
from slp.config.templates.training import TrainingConfig
from slp.data.datasets.isolated_supervised import IsolatedSignsRecognitionDataset
from slp.losses.loading import load_multihead_criterion
from slp.metrics.classification.base import ClassificationMetrics
from slp.tasks.isolated_recognition.model import load_model
from slp.trainers.base import TrainerBase
from slp.utils.model import count_parameters
from slp.schedulers.cosine_annealing_with_linear_warmup import create_warmup_plateau_cosine_scheduler


class IsolatedRecognitionTrainer(TrainerBase):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        n_classes: int,
        is_output_multilayer: bool,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.is_output_multilayer = is_output_multilayer

        self.training_metrics = ClassificationMetrics(prefix="training/cls/", n_classes=n_classes)
        self.validation_metrics = ClassificationMetrics(prefix="validation/cls/", n_classes=n_classes)
        self.testing_metrics = ClassificationMetrics(prefix="testing/cls/", n_classes=n_classes)

        self.test_logits = {}
        self.save_hyperparameters(ignore=["model", "criterion", "test_logits"])

    def prediction_step(self, batch, mode):
        features, masks, targets = batch["poses"], batch["masks"], batch["label"]
        batch_size = features.size(0)
        # features of shape (N, C_in, T)
        # mask of shape (N, 1, T)
        # classification target of shape N

        logits = self.model(features, masks)
        loss = self.criterion(logits, {'classification': logits['classification']})

        self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        cls_logits = logits["classification"]
        cls_logits = cls_logits[-1] if self.is_output_multilayer else cls_logits
        per_frame_probs = cls_logits.softmax(dim=1)
        if mode == "training":
            metrics = self.training_metrics(per_frame_probs, targets)
        elif mode == "validation":
            metrics = self.validation_metrics(per_frame_probs, targets)
        elif mode == "testing":
            metrics = self.testing_metrics(per_frame_probs, targets)
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
        instance_ids, lengths, targets = batch["id"], batch["length"], batch["label"]
        logits, loss, metrics = self.prediction_step(batch, "testing")
        if self.is_output_multilayer:
            logits = {k: v[-1] for k, v in logits.items()}

        # batch_size = len(instance_ids)
        # for idx in range(batch_size):
        #     instance_id = instance_ids[idx]
        #     self.test_logits[f"{instance_id}_{starts[idx]}_{ends[idx]}"] = {
        #         head_name: head_logits[idx]
        #         .detach()
        #         .cpu()
        #         .numpy()[..., : lengths[idx]]
        #         .astype("float16")
        #         for head_name, head_logits in logits.items()
        #     }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = create_warmup_plateau_cosine_scheduler(
            optimizer,
            n_warmup_steps=20,
            n_plateau_steps=10,
            max_steps=150,
            lr=self.learning_rate,
            start_lr=1e-12,
            end_lr=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_isolated_recognition_trainer(
    training_dataset: IsolatedSignsRecognitionDataset,
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
    cls_head = model_config.heads.get("classification")
    if cls_head is None:
        raise ValueError("An isolated recognition model must have a 'classification' head.")
    model = load_model(model_config)
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")
    checkpoint_path = model_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return IsolatedRecognitionTrainer.load_from_checkpoint(
            checkpoint_path, model=model, criterion=criterion
        )
    return IsolatedRecognitionTrainer(
        model=model,
        criterion=criterion,
        learning_rate=training_config.learning_rate,
        n_classes=cls_head.out_channels,
        is_output_multilayer=training_config.is_output_multilayer,
    )
