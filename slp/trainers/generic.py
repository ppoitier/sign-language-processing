from torch import nn, optim, Tensor

from slp.trainers.base import TrainerBase


class GenericTrainer(TrainerBase):
    """Base trainer for multi-head models.

    Handles the common training loop: forward pass, loss computation,
    logging, optimizer setup, multi-layer output extraction, and test-time
    logit caching.

    Subclasses must implement:
        - ``compute_metrics``: task-specific metric computation.

    Subclasses may override:
        - ``on_test_batch``: additional test-time logic (e.g. segment decoding).
        - ``cache_test_logits``: custom logit caching strategy.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        heads_to_targets: dict[str, str],
        is_output_multilayer: bool = False,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.heads_to_targets = heads_to_targets
        self.is_output_multilayer = is_output_multilayer

        self.test_logits: dict = {}
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward_step(self, batch: dict) -> tuple[dict, Tensor, dict, int]:
        """Run model forward and compute loss.

        Returns:
            raw_logits: dict of head_name -> logits (possibly multi-layer).
            loss: scalar total loss.
            task_losses: dict of task_name -> individual task loss.
            batch_size: int.
        """
        features, masks, targets = batch["poses"], batch["masks"], batch["targets"]
        batch_size = features.size(0)
        features = features.permute(0, 2, 1).float().contiguous()
        masks = masks.unsqueeze(1).bool().contiguous()

        raw_logits = self.model(features, masks)
        losses = self.criterion(
            raw_logits,
            {
                head_name: targets[target_name]
                for head_name, target_name in self.heads_to_targets.items()
            },
        )
        loss = losses["total_loss"]
        return raw_logits, loss, losses, batch_size

    def extract_eval_logits(self, raw_logits: dict) -> dict:
        """Take the last layer from each head if the model is multi-layer."""
        if self.is_output_multilayer:
            return {k: v[-1] for k, v in raw_logits.items()}
        return raw_logits

    def compute_metrics(self, logits: dict, batch: dict, mode: str) -> dict:
        """Compute task-specific metrics. Must be implemented by subclasses.

        Args:
            logits: eval-ready logits (last layer already extracted).
            batch: the full batch dict from the dataloader.
            mode: one of "training", "validation", "testing".

        Returns:
            A dict of metric_name -> value to be logged.
        """
        raise NotImplementedError

    def on_test_batch(self, logits: dict, batch: dict, batch_size: int) -> None:
        """Called during test_step after metrics. Override for task-specific
        test logic such as segment decoding. Default is a no-op."""
        pass

    def cache_test_logits(self, logits: dict, batch: dict) -> None:
        """Store per-instance logits for later analysis.

        Override if your batch structure uses different keys for instance
        identification or if you need a different storage format.
        """
        instance_ids = batch["id"]
        starts = batch["start"]
        ends = batch["end"]
        lengths = batch["lengths"]

        for idx in range(len(instance_ids)):
            key = f"{instance_ids[idx]}_{starts[idx]}_{ends[idx]}"
            self.test_logits[key] = {
                head_name: head_logits[idx]
                .detach()
                .cpu()
                .numpy()[..., : lengths[idx]]
                .astype("float16")
                for head_name, head_logits in logits.items()
            }

    def prediction_step(self, batch: dict, mode: str):
        raw_logits, loss, task_losses, batch_size = self.forward_step(batch)
        self.log(
            f"{mode}/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size
        )
        for task_name, task_loss in task_losses.items():
            self.log(
                f"{mode}/{task_name}_loss",
                task_loss,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        eval_logits = self.extract_eval_logits(raw_logits)
        metrics = self.compute_metrics(eval_logits, batch, mode)
        self.log_metrics(metrics, batch_size=batch_size)

        return raw_logits, eval_logits, loss, batch_size

    def training_step(self, batch, batch_idx):
        _, _, loss, _ = self.prediction_step(batch, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        self.prediction_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        _, eval_logits, _, batch_size = self.prediction_step(batch, "testing")
        self.cache_test_logits(eval_logits, batch)
        self.on_test_batch(eval_logits, batch, batch_size)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
