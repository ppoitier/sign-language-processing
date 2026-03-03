from torch.nn import Identity

from slp.core.registry import HEAD_REGISTRY

HEAD_REGISTRY.register("identity")(Identity)
