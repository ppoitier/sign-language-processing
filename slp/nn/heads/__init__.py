from torch.nn import Identity

from slp.core.registry import HEAD_REGISTRY
import slp.nn.heads.linear
import slp.nn.heads.trident

HEAD_REGISTRY.register("identity")(Identity)
