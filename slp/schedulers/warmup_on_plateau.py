import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        start_lr,
        base_lr,
        **plateau_kwargs,
    ):
        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_lr / base_lr if base_lr > 0 else 0,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_done = False

        super().__init__(optimizer, **plateau_kwargs)

    def step(self, metrics=None, epoch=None):
        if self.warmup_scheduler.last_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
            if self.warmup_scheduler.last_epoch >= self.warmup_epochs:
                self.warmup_done = True
                # Sync ReduceLROnPlateau's internal tracking of LR
                self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        else:
            if metrics is None:
                warnings.warn(
                    "WarmupReduceLROnPlateau expects metrics (e.g. val_loss) after warmup.",
                    UserWarning,
                )
            # Standard ReduceLROnPlateau behavior
            super().step(metrics)

    def state_dict(self):
        """Return state of both schedulers."""
        state = super().state_dict()
        state["warmup_scheduler"] = self.warmup_scheduler.state_dict()
        state["warmup_done"] = self.warmup_done
        return state

    def load_state_dict(self, state_dict):
        """Load state for both schedulers."""
        self.warmup_scheduler.load_state_dict(state_dict.pop("warmup_scheduler"))
        self.warmup_done = state_dict.pop("warmup_done")
        super().load_state_dict(state_dict)
