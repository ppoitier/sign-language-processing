from typing import Optional, Sequence

from torch import nn, Tensor

from slp.core.config.training import RepresentationLearningTrainingConfig
from slp.utils.model import count_parameters

from slp.trainers.base import TrainerBase
from slp.nn.ssl.base import SSLEncoder, SSLMethod
from slp.nn.ssl.loading import build_ssl_method
from slp.nn.ssl.stats import representation_statistics
from slp.schedulers.types import OptimizerFactory, SchedulerFactory


class RepresentationLearningTrainer(TrainerBase):
    """Trainer for self-supervised representation learning.

    It is the counterpart of ``GenericTrainer`` for label-free objectives: the
    batch carries several augmented views of each sample instead of targets,
    and the objective is a pluggable ``SSLMethod`` (SimCLR, SimSiam, BYOL, ...)
    rather than a per-head criterion.

    The trainer stays method-agnostic. It only:
        - unpacks the views from the batch and checks their count,
        - hands them to the method along with the encoder,
        - logs the losses, the method's own metrics and collapse diagnostics,
        - forwards the post-optimizer-step hook the momentum methods need.

    Adding a new method therefore never requires touching this class.

    Args:
        model: the encoder, typically a ``HydraModel`` whose ``embedding_head``
            is a ``projection`` head. Other heads are computed but ignored,
            unless the method asks for them.
        method: the self-supervised objective.
        learning_rate: used when no ``optimizer_factory`` is given.
        embedding_head: name of the head carrying the projected embedding.
        is_output_multistage: kept for symmetry with ``GenericTrainer``; the
            encoder always takes the last stage, since intermediate stages have
            no meaning for a pooled embedding.
        log_representation_stats: log collapse diagnostics. Cheap, and the only
            reliable signal that pretraining is working, so on by default.
        cache_test_embeddings: keep per-instance embeddings during ``test`` in
            ``self.test_embeddings``, for downstream probing or clustering.
    """

    def __init__(
        self,
        model: nn.Module,
        method: SSLMethod,
        learning_rate: float,
        embedding_head: str = "projection",
        is_output_multistage: bool = False,
        log_representation_stats: bool = True,
        cache_test_embeddings: bool = False,
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
        scheduler_interval: str = "epoch",
        scheduler_monitor: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = SSLEncoder(
            model=model,
            embedding_head=embedding_head,
            is_output_multistage=is_output_multistage,
        )
        self.method = method
        # Lets the method build encoder-derived modules (e.g. BYOL's EMA copy)
        # before the optimizer collects parameters.
        self.method.setup_method(self.encoder)

        self.setup_optimization(
            learning_rate=learning_rate,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_interval=scheduler_interval,
            scheduler_monitor=scheduler_monitor,
        )

        self.log_representation_stats = log_representation_stats
        self.cache_test_embeddings = cache_test_embeddings
        self.test_embeddings: dict = {}

        self.save_hyperparameters(
            ignore=[
                "model",
                "method",
                "test_embeddings",
                "optimizer_factory",
                "scheduler_factory",
            ]
        )

    @property
    def model(self) -> nn.Module:
        """The wrapped encoder model, for symmetry with the other trainers."""
        return self.encoder.model

    def optimized_parameters(self):
        """Skip frozen parameters, such as BYOL's target encoder."""
        return (p for p in self.parameters() if p.requires_grad)

    # # I think this function should be removed. It's unnecessary
    # def extract_views(self, batch: dict) -> tuple[list[Tensor], list[Optional[Tensor]]]:
    #     """Unpack the augmented views and their masks from a batch.
    #
    #     ``batch["poses"]`` is expected to be a tuple of view tensors. A single
    #     tensor is accepted too, which is what makes single-view methods (and
    #     debugging on a supervised loader) work unchanged.
    #     """
    #     views = batch["poses"]
    #     if isinstance(views, Tensor):
    #         views = (views,)
    #     views = list(views)
    #
    #     masks = batch.get("masks")
    #     if masks is None:
    #         masks = [None] * len(views)
    #     elif isinstance(masks, Tensor):
    #         # One shared mask: the views keep the original temporal layout.
    #         masks = [masks] * len(views)
    #     else:
    #         masks = list(masks)
    #
    #     if len(masks) != len(views):
    #         raise ValueError(
    #             f"Got {len(views)} views but {len(masks)} masks. The batch must "
    #             f"provide either one mask per view or a single shared mask."
    #         )
    #
    #     expected = self.method.n_views
    #     if len(views) != expected:
    #         raise ValueError(
    #             f"{type(self.method).__name__} expects {expected} views per sample, "
    #             f"but the batch provides {len(views)}. Adjust the augmentation "
    #             f"pipeline or the method's 'n_views'."
    #         )
    #
    #     return views, masks

    def prediction_step(self, batch: dict, mode: str):
        # batch['poses'] is already a tuple of tensors of shape (N, T, C_in)
        # batch['masks'] is already a tuple of boolean tensors of shape (N, T)

        views, masks = batch['poses'], batch['masks']
        batch_size = views[0].size(0)

        output = self.method(self.encoder, views, masks)

        loss = output.losses["total_loss"]
        self.log(
            f"{mode}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        for loss_name, loss_value in output.losses.items():
            if loss_name == "total_loss":
                continue
            self.log(
                f"{mode}/{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        if output.metrics:
            self.log_metrics(
                {f"{mode}/{name}": value for name, value in output.metrics.items()},
                batch_size=batch_size,
            )

        if self.log_representation_stats and output.embeddings:
            self.log_metrics(
                representation_statistics(output.embeddings, prefix=f"{mode}/"),
                batch_size=batch_size,
            )

        return output, loss, batch_size

    def training_step(self, batch, batch_idx):
        _, loss, _ = self.prediction_step(batch, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        self.prediction_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        output, _, _ = self.prediction_step(batch, "testing")
        if self.cache_test_embeddings:
            self.cache_embeddings(output.embeddings, batch)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Drive momentum updates, after the optimizer step rather than before."""
        max_steps = None
        if self.trainer is not None:
            estimated = self.trainer.estimated_stepping_batches
            # Lightning returns inf when it cannot estimate the total.
            max_steps = int(estimated) if estimated not in (None, float("inf")) else None
        self.method.on_train_step_end(step=self.global_step, max_steps=max_steps)

    def cache_embeddings(self, embeddings: Sequence[Tensor], batch: dict) -> None:
        """Store the first view's embedding per instance, keyed like the logits
        cached by the supervised trainers so the same tooling can read them."""
        if not embeddings:
            return
        instance_ids = batch["id"]
        first_view = embeddings[0].detach().cpu().numpy().astype("float16")
        for idx in range(len(instance_ids)):
            self.test_embeddings[str(instance_ids[idx])] = first_view[idx]


def load_representation_learning_trainer(
    model: nn.Module,
    training_config: RepresentationLearningTrainingConfig,
    method: Optional[SSLMethod] = None,
    optimizer_factory: Optional[OptimizerFactory] = None,
    scheduler_factory: Optional[SchedulerFactory] = None,
    scheduler_interval: str = "epoch",
    scheduler_monitor: Optional[str] = None,
) -> RepresentationLearningTrainer:
    n_parameters = count_parameters(model)
    print(f"Total number of parameters in the model: {n_parameters:,}")

    if method is None:
        method = build_ssl_method(training_config.method, embedding_dim=training_config.embedding_dim)
    print(f"Using self-supervised method: {type(method).__name__}")

    checkpoint_path = training_config.checkpoint_path
    if checkpoint_path is not None:
        print("Loading checkpoint:", checkpoint_path)
        return RepresentationLearningTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            method=method,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_interval=scheduler_interval,
            scheduler_monitor=scheduler_monitor,
            weights_only=False,
        )

    return RepresentationLearningTrainer(
        model=model,
        method=method,
        learning_rate=training_config.learning_rate,
        embedding_head=training_config.embedding_head,
        is_output_multistage=training_config.is_output_multistage,
        log_representation_stats=training_config.log_representation_stats,
        cache_test_embeddings=training_config.cache_test_embeddings,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        scheduler_interval=scheduler_interval,
        scheduler_monitor=scheduler_monitor,
    )
