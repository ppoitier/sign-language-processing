from slp.config.templates.data import SegmentCodecConfig
from slp.codecs.annotations.actionness import ActionnessCodec
from slp.codecs.annotations.boundaries import BoundariesCodec
from slp.codecs.annotations.offsets import OffsetsCodec


def load_annotation_codec(config: SegmentCodecConfig):
    match config.name:
        case 'actionness':
            return ActionnessCodec(**config.args)
        case 'boundaries':
            return BoundariesCodec(**config.args)
        case 'offsets':
            return OffsetsCodec(**config.args)
        case _:
            raise ValueError(f'Unknown annotation codec: {config.name}.')
