from torch import optim


class WarmupThenReduceOnPlateau:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        n_warmup_steps: int,
        lr: float,
        start_lr: float,
        reduce_factor: float,
        reduce_patience: int,
    ):
        self.n_warmup_steps = n_warmup_steps
        self.step_count = 0
        self.warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_lr / lr if lr > 0 else 0,
            total_iters=n_warmup_steps,
        )
        self.plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=reduce_factor, patience=reduce_patience
        )

    def step(self, metric=None):
        if self.step_count < self.n_warmup_steps:
            self.warmup.step()
        else:
            # ReduceLROnPlateau needs a metric
            if metric is not None:
                self.plateau.step(metric)
        self.step_count += 1