from slp.codecs.segmentation import PerFrameLabelsCodec, BoundariesCodec, BioTaggingCodec, OffsetsCodec
from slp.config.templates.data import SegmentCodecConfig


def load_segments_codec(config: SegmentCodecConfig):
    if config.name.endswith('+offsets'):
        scoring_codec = load_segments_codec(SegmentCodecConfig(name=config.name[:-len('+offsets')]))
        return OffsetsCodec(scoring_codec=scoring_codec, **config.args)
    elif config.name == 'actionness':
        return PerFrameLabelsCodec(**config.args)
    elif config.name == 'boundaries':
        return BoundariesCodec(**config.args)
    elif config.name == 'bio_tags':
        return BioTaggingCodec(**config.args)
    raise ValueError(f'Segment codec {config.name} not found.')
