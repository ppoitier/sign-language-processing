import copy
import math
from typing import Optional

import torch
from torch import nn


class MomentumEncoder(nn.Module):
    """A frozen exponential-moving-average copy of an online module.

    Used by BYOL, MoCo and their descendants as the "target" network.
    Registered as a sub-module so the EMA weights are checkpointed and a run
    can be resumed exactly; its parameters have ``requires_grad=False`` and are
    therefore filtered out of the optimizer.

    Args:
        online: the module to track. Deep-copied once at construction.
        base_momentum: EMA coefficient at step 0.
        final_momentum: coefficient the momentum anneals towards. BYOL ramps
            from 0.996 to 1.0 on a cosine schedule; set it equal to
            ``base_momentum`` (the default) for MoCo's constant momentum.
    """

    def __init__(
        self,
        online: nn.Module,
        base_momentum: float = 0.996,
        final_momentum: Optional[float] = None,
    ):
        super().__init__()
        self.module = copy.deepcopy(online)
        self.module.requires_grad_(False)

        self.base_momentum = base_momentum
        self.final_momentum = (
            base_momentum if final_momentum is None else final_momentum
        )
        self.current_momentum = base_momentum

    def train(self, mode: bool = True):
        # The target network keeps running in train mode so its BatchNorm
        # statistics track the current data distribution, as in BYOL/MoCo.
        return super().train(mode)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)

    def momentum_at(self, step: int, max_steps: Optional[int]) -> float:
        """Cosine ramp from ``base_momentum`` to ``final_momentum``."""
        if self.final_momentum == self.base_momentum or not max_steps:
            return self.base_momentum
        progress = min(max(step / max_steps, 0.0), 1.0)
        decay = (math.cos(math.pi * progress) + 1) / 2
        return self.final_momentum - (self.final_momentum - self.base_momentum) * decay

    @torch.no_grad()
    def update(
        self,
        online: nn.Module,
        step: int = 0,
        max_steps: Optional[int] = None,
    ) -> float:
        """Pull the target weights towards the online ones.

        Buffers (BatchNorm running stats) are copied rather than averaged,
        which is what the reference implementations do.
        """
        momentum = self.momentum_at(step, max_steps)
        self.current_momentum = momentum

        for target_param, online_param in zip(
            self.module.parameters(), online.parameters()
        ):
            target_param.mul_(momentum).add_(online_param.detach(), alpha=1 - momentum)

        for target_buffer, online_buffer in zip(
            self.module.buffers(), online.buffers()
        ):
            target_buffer.copy_(online_buffer)

        return momentum
