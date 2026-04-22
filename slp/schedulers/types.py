from typing import Callable, Iterable

from torch import optim


OptimizerFactory = Callable[[Iterable], optim.Optimizer]
SchedulerFactory = Callable[[optim.Optimizer], optim.lr_scheduler.LRScheduler]