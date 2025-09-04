from slp.config.templates.data import SegmentCodecConfig
from slp.codecs.annotations.actionness import ActionnessCodec
from slp.codecs.annotations.boundaries import BoundariesCodec
from slp.codecs.annotations.offsets import OffsetsCodec


def load_annotation_codec(config: SegmentCodecConfig):
    match config.name:
        case 'actionness':
            return ActionnessCodec()
        case 'boundaries':
            return BoundariesCodec()
        case 'offsets':
            return OffsetsCodec()
        case _:
            raise ValueError(f'Unknown annotation codec: {config.name}.')
