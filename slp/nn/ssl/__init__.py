from slp.nn.ssl.base import SSLEncoder, SSLMethod, SSLOutput
from slp.nn.ssl.momentum import MomentumEncoder
from slp.nn.ssl.predictor import PredictionHead
from slp.nn.ssl.stats import representation_statistics

# Imported for their registration side effect.
# noinspection PyUnresolvedReferences
from slp.nn.ssl.simclr import SimCLR

# noinspection PyUnresolvedReferences
from slp.nn.ssl.simsiam import SimSiam

# noinspection PyUnresolvedReferences
from slp.nn.ssl.byol import BYOL

__all__ = [
    "SSLEncoder",
    "SSLMethod",
    "SSLOutput",
    "MomentumEncoder",
    "PredictionHead",
    "representation_statistics",
    "SimCLR",
    "SimSiam",
    "BYOL",
]
