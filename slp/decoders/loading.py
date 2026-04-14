from slp.core.registry import SEGMENT_DECODER_REGISTRY
from slp.core.config.training import SegmentDecoderConfig


def load_segment_decoder(config: SegmentDecoderConfig):
    decoder_cls = SEGMENT_DECODER_REGISTRY.get(config.name)
    return decoder_cls(**config.kwargs)
