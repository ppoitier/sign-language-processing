from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from slp.core.registry import SSL_METHOD_REGISTRY
from slp.nn.ssl.base import SSLEncoder, SSLMethod, SSLOutput
from slp.nn.ssl.momentum import MomentumEncoder
from slp.nn.ssl.predictor import PredictionHead
from slp.nn.ssl.simsiam import normalized_output_std


@SSL_METHOD_REGISTRY.register("byol")
class BYOL(SSLMethod):
    """BYOL: predict the output of a momentum-averaged copy of the encoder.

    Like SimSiam it uses a predictor and no negatives, but the regression
    target comes from a separate target network whose weights are an
    exponential moving average of the online ones, with the momentum ramped
    towards 1 over training (Grill et al., 2020).

    The target encoder is built in ``setup_method`` because it is a copy of
    the encoder, which does not exist yet at construction time.

    Args:
        embedding_dim: output dimension of the model's projection head.
        predictor_hidden_dim: bottleneck width of the predictor.
        base_momentum: EMA coefficient at the first step.
        final_momentum: coefficient reached at the end of training. Set it
            equal to ``base_momentum`` to disable the ramp.
        n_views: number of augmented views per sample.
    """

    def __init__(
        self,
        embedding_dim: int,
        predictor_hidden_dim: int | None = None,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
        n_views: int = 2,
    ):
        super().__init__()
        if n_views < 2:
            raise ValueError(f"BYOL needs at least 2 views, got {n_views}.")
        self.n_views = n_views
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum

        self.predictor = PredictionHead(
            in_channels=embedding_dim,
            hidden_channels=predictor_hidden_dim,
            out_channels=embedding_dim,
        )
        self.target_encoder: MomentumEncoder | None = None
        self._online_encoder: SSLEncoder | None = None

    def setup_method(self, encoder: SSLEncoder) -> None:
        self.target_encoder = MomentumEncoder(
            online=encoder,
            base_momentum=self.base_momentum,
            final_momentum=self.final_momentum,
        )
        # Kept out of the module tree (leading underscore is not enough, so we
        # stash it in __dict__) to avoid registering the online encoder twice.
        object.__setattr__(self, "_online_encoder", encoder)

    def forward(
        self,
        encoder: SSLEncoder,
        views: Sequence[Tensor],
        masks: Sequence[Optional[Tensor]],
    ) -> SSLOutput:
        if self.target_encoder is None:
            raise RuntimeError(
                "BYOL.setup_method() must be called before the first forward pass."
            )

        online = [encoder.embed(view, mask) for view, mask in zip(views, masks)]
        predictions = [self.predictor(z) for z in online]

        with torch.no_grad():
            targets = [
                self.target_encoder.module.embed(view, mask).detach()
                for view, mask in zip(views, masks)
            ]

        losses = []
        for i, prediction in enumerate(predictions):
            for j, target in enumerate(targets):
                if i == j:
                    continue
                # 2 - 2*cos, the squared error between L2-normalised vectors.
                losses.append(
                    2 - 2 * F.cosine_similarity(prediction, target, dim=-1).mean()
                )
        loss = torch.stack(losses).mean()

        return SSLOutput(
            losses={"total_loss": loss},
            embeddings=online,
            metrics={
                "output_std": normalized_output_std(online),
                "momentum": torch.tensor(
                    self.target_encoder.current_momentum, device=loss.device
                ),
            },
        )

    def on_train_step_end(self, step: int, max_steps: Optional[int] = None) -> None:
        if self.target_encoder is None or self._online_encoder is None:
            return
        self.target_encoder.update(
            online=self._online_encoder, step=step, max_steps=max_steps
        )
