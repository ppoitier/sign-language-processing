from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    ConstantLR,
)


class CosineAnnealingWithLinearWarmup(LRScheduler):
    def __init__(
            self,
            optimizer,
            n_warmup_steps: int,
            max_steps: int,
            lr: float,
            start_lr: float,
            end_lr: float,
            last_epoch=-1,
    ):
        self.n_warmup_steps = n_warmup_steps
        self.max_steps = max_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=start_lr / lr, total_iters=self.n_warmup_steps),
                CosineAnnealingLR(optimizer, T_max=self.max_steps - self.n_warmup_steps, eta_min=self.end_lr),
            ],
            last_epoch=last_epoch,
            milestones=[n_warmup_steps],
        )
        super().__init__(optimizer, last_epoch)

    def step(self):
        self.scheduler.step()
        super().step()

    def get_lr(self):
        return self.scheduler.get_last_lr()


def create_warmup_plateau_cosine_scheduler(
    optimizer: Optimizer,
    n_warmup_steps: int,
    n_plateau_steps: int,  # <-- Highlighted Change: New parameter
    max_steps: int,
    lr: float,
    start_lr: float,
    end_lr: float,
):
    """
    Creates a scheduler with linear warmup, a constant plateau, and cosine annealing.
    """
    # 1. Warmup Phase
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_lr / lr if lr > 0 else 0,
        total_iters=n_warmup_steps,
    )

    # 2. Plateau Phase
    # The factor is 1.0 because we want to hold the peak learning rate.
    plateau_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=n_plateau_steps)

    # 3. Cosine Decay Phase
    cosine_steps = max_steps - n_warmup_steps - n_plateau_steps
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=end_lr)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, plateau_scheduler, cosine_scheduler],
        milestones=[n_warmup_steps, n_warmup_steps + n_plateau_steps],
    )
    return scheduler
