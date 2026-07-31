from dataclasses import dataclass, field
from typing import Optional, Sequence

from torch import nn, Tensor


class SSLEncoder(nn.Module):
    """Adapts a framework model (typically a ``HydraModel``) to the flat
    interface that self-supervised methods expect.

    It absorbs the input-layout fiddling that ``GenericTrainer.forward_step``
    does (permute, dtype, mask unsqueeze) and the multi-stage output
    extraction, so that a method only ever sees ``(B, E)`` embeddings.

    The model is expected to expose a head producing the projected embedding
    (``embedding_head``). If that head still carries a temporal axis, it is
    average-pooled — but pooling really belongs in the head itself (see the
    ``projection`` head), so this is only a fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_head: str = "projection",
        is_output_multistage: bool = False,
    ):
        super().__init__()
        self.model = model
        self.embedding_head = embedding_head
        self.is_output_multistage = is_output_multistage

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Returns one tensor per head, with the multi-stage axis removed."""
        if x.ndim == 3:
            # Permute to fit CNN dimensions when temporal features.
            x = x.permute(0, 2, 1)
        x = x.float().contiguous()
        if mask is not None:
            mask = mask.unsqueeze(1).bool().contiguous()

        raw_outputs = self.model(x, mask)

        outputs = {}
        for head_name, head_output in raw_outputs.items():
            if isinstance(head_output, (list, tuple)):
                # HydraModel always returns a list, one entry per stage.
                head_output = head_output[-1]
            outputs[head_name] = head_output
        return outputs

    def embed(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Projected embedding of a single view, of shape ``(B, E)``."""
        outputs = self(x, mask)
        if self.embedding_head not in outputs:
            raise KeyError(
                f"Embedding head '{self.embedding_head}' not found in the model "
                f"outputs. Available heads: {sorted(outputs.keys())}."
            )
        embedding = outputs[self.embedding_head]
        if embedding.ndim > 2:
            # Fallback pooling: the projection head should already pool.
            embedding = embedding.flatten(2).mean(-1)
        return embedding


@dataclass
class SSLOutput:
    """What a method returns for one batch.

    Attributes:
        losses: named losses, and must contain ``"total_loss"`` (the one that
            is backpropagated). Extra entries are logged as ``<mode>/<name>``.
        embeddings: the online embeddings of each view, ``(B, E)``. Only used
            for representation diagnostics, so a method may return an empty
            list to skip them.
        metrics: optional method-specific scalars to log (already detached).
    """

    losses: dict[str, Tensor]
    embeddings: list[Tensor] = field(default_factory=list)
    metrics: dict[str, Tensor] = field(default_factory=dict)


class SSLMethod(nn.Module):
    """Base class for self-supervised representation learning objectives.

    A method owns whatever extra parameters its objective needs (a predictor
    for SimSiam, an EMA target encoder for BYOL, a queue for MoCo), which is
    why it is an ``nn.Module`` rather than a plain loss: those parameters must
    live in the trainer's ``state_dict`` and, when trainable, in the optimizer.

    Lifecycle, driven by ``RepresentationLearningTrainer``:
        1. ``__init__``  — build objective-local modules that need no encoder.
        2. ``setup_method(encoder)`` — build modules derived from the encoder
           (e.g. an EMA copy of it). Called once, before training.
        3. ``forward(encoder, views, masks)`` — every step.
        4. ``on_train_step_end(step, max_steps)`` — after the optimizer step,
           for momentum updates and schedules.

    Subclasses must implement ``forward``, and set ``n_views`` if they need
    something other than two views.
    """

    #: Number of augmented views the method expects per sample.
    n_views: int = 2

    def setup_method(self, encoder: SSLEncoder) -> None:
        """Hook to build encoder-derived modules. Default is a no-op."""
        pass

    def forward(
        self,
        encoder: SSLEncoder,
        views: Sequence[Tensor],
        masks: Sequence[Optional[Tensor]],
    ) -> SSLOutput:
        """Compute the objective for one batch of views.

        The method drives the encoder itself (rather than receiving embeddings)
        so that asymmetric methods can run extra forward passes, such as BYOL
        pushing the views through its frozen target network.

        Args:
            encoder: the online encoder; ``encoder.embed(view, mask)`` gives
                a ``(B, E)`` embedding.
            views: ``n_views`` augmented versions of the same batch.
            masks: the matching padding masks, entries may be ``None``.
        """
        raise NotImplementedError

    def on_train_step_end(self, step: int, max_steps: Optional[int] = None) -> None:
        """Called after each optimizer step. Default is a no-op.

        Args:
            step: the trainer's global step.
            max_steps: total planned optimizer steps, when the trainer can
                estimate it. Used by momentum/temperature schedules.
        """
        pass

    def trainable_parameters(self):
        """Parameters of this method that the optimizer should update.

        Frozen modules (EMA targets) are excluded by the ``requires_grad``
        filter, so subclasses rarely need to override this.
        """
        return (p for p in self.parameters() if p.requires_grad)
