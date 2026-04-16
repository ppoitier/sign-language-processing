from torch.nn import Identity, Linear

from slp.core.registry import HEAD_REGISTRY
import slp.nn.heads.linear

HEAD_REGISTRY.register("identity")(Identity)
