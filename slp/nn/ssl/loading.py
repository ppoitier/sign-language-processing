from inspect import signature
from typing import Optional

from slp.core.config.model import HydraConfig
from slp.core.config.training import SSLMethodConfig
from slp.core.registry import SSL_METHOD_REGISTRY

# Imported for their registration side effect.
# noinspection PyUnusedImports
import slp.nn.ssl.simclr

# noinspection PyUnusedImports
import slp.nn.ssl.simsiam

# noinspection PyUnusedImports
import slp.nn.ssl.byol

from slp.nn.ssl.base import SSLMethod


def build_ssl_method(
    config: SSLMethodConfig,
    embedding_dim: Optional[int] = None,
) -> SSLMethod:
    """Instantiates a self-supervised objective from its configuration.

    Args:
        config: the method name and its kwargs.
        embedding_dim: injected as ``embedding_dim`` when the config does not
            set it, so that predictor-based methods do not have to repeat the
            projector's output size. See ``infer_embedding_dim``.
    """
    method_cls = SSL_METHOD_REGISTRY.get(config.name)
    kwargs = dict(config.kwargs)
    # Only methods that own a predictor take an embedding_dim; SimCLR does not.
    if embedding_dim is not None and "embedding_dim" in signature(method_cls).parameters:
        kwargs.setdefault("embedding_dim", embedding_dim)
    return method_cls(**kwargs)


def infer_embedding_dim(
    config: HydraConfig,
    embedding_head: str = "projection",
) -> Optional[int]:
    """Reads the embedding size off the projection head of a model config.

    Returns ``None`` when the head does not declare ``out_features``, in which
    case the method must be given its ``embedding_dim`` explicitly.
    """
    head_config = config.heads.get(embedding_head)
    if head_config is None:
        return None
    return head_config.kwargs.get("out_features")
