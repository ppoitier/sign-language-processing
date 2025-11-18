from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    ConstantLR,
    ReduceLROnPlateau,
)


def create_warmup_reduce_on_plateau(
        optimizer: Optimizer,
        n_warmup_steps: int,
        lr: float,
        start_lr: float,
        reduce_factor: float,
        reduce_patience: int,
):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_lr / lr if lr > 0 else 0,
        total_iters=n_warmup_steps,
    )
    reduce_on_plateau = ReduceLROnPlateau(optimizer, factor=reduce_factor, patience=reduce_patience)
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, reduce_on_plateau],
        milestones=[n_warmup_steps],
    )


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
