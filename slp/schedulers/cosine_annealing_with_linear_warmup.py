from typing import Optional

from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, LinearLR, SequentialLR


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
