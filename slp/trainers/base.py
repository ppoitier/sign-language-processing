from torch import Tensor
import lightning as pl


class TrainerBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_metrics(
        self,
        metrics: dict[str, any],
        batch_size: int,
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        for name, value in metrics.items():
            if isinstance(value, Tensor) and value.numel() > 1:
                assert len(value.shape) == 1
                for idx, v in enumerate(value.tolist()):
                    self.log(
                        f"{name}/{idx}",
                        v,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        batch_size=batch_size,
                    )
            else:
                self.log(
                    name,
                    value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )