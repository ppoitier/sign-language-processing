import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_epochs, **plateau_kwargs):
        """
        Args:
            optimizer: The optimizer from the LightningModule.
            warmup_epochs: The number of epochs for the linear warmup phase.
            **plateau_kwargs: Keyword arguments for the underlying ReduceLROnPlateau scheduler
                              (e.g., mode, factor, patience).
        """
        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, **plateau_kwargs)

    def step(self, metrics=None, epoch=None):
        """
        The step function that PyTorch Lightning calls at the end of each epoch.
        It delegates to the appropriate scheduler based on the current epoch.
        """
        # self.warmup_scheduler.last_epoch starts at -1 and is incremented with each step.
        # Warmup phase is for epochs 0 to (warmup_epochs - 1).
        if self.warmup_scheduler.last_epoch < self.warmup_epochs - 1:
            # During warmup, step the warmup scheduler
            self.warmup_scheduler.step()
        else:
            # After warmup, step the parent ReduceLROnPlateau scheduler
            if metrics is None:
                # This should not happen with a correctly configured Lightning Trainer
                warnings.warn(
                    "The `WarmupReduceLROnPlateau` scheduler expects a metric to be passed to `step()` after the warmup phase.",
                    UserWarning,
                )

            # The 'step' method of the parent class (ReduceLROnPlateau) requires the metric.
            super().step(metrics)
