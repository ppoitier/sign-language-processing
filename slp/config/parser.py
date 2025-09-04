from pathlib import Path

import yaml

from slp.utils.dict import deep_merge


def _parse_config_to_dict(yaml_filepath: str | Path) -> dict:
    if isinstance(yaml_filepath, str):
        yaml_filepath = Path(yaml_filepath)
    with open(yaml_filepath, 'r') as stream:
        config = yaml.safe_load(stream)
    if 'from' in config:
        parent_path = (yaml_filepath.parent / Path(config['from'])).resolve()
        parent_config = _parse_config_to_dict(parent_path)
        parent_config = deep_merge(config, parent_config)
        return parent_config
    return config


def parse_config(yaml_filepath: str | Path, pydantic_class):
    config_dict = _parse_config_to_dict(yaml_filepath)
    return pydantic_class.model_validate(config_dict)

