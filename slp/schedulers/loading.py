from typing import Optional

from slp.core.registry import LR_SCHEDULER_REGISTRY
from slp.core.config.training import LRSchedulerConfig
from slp.schedulers.types import SchedulerFactory


def load_lr_scheduler_factory(config: LRSchedulerConfig) -> tuple[SchedulerFactory, Optional[str]]:
    scheduler_cls = LR_SCHEDULER_REGISTRY.get(config.name)
    scheduler_factory = lambda optimizer: scheduler_cls(optimizer, **config.kwargs)
    return scheduler_factory, config.monitor
