from pathlib import Path

import yaml

from slp.utils.dict import deep_merge
from slp.config.templates.task import SegmentationTaskConfig, ContrastiveRecognitionTask, RecognitionTaskConfig


def _load_config_to_dict(yaml_filepath: str | Path) -> dict:
    if isinstance(yaml_filepath, str):
        yaml_filepath = Path(yaml_filepath)
    with open(yaml_filepath, 'r') as stream:
        config = yaml.safe_load(stream)
    if 'from' in config:
        parent_path = (yaml_filepath.parent / Path(config['from'])).resolve()
        parent_config = _load_config_to_dict(parent_path)
        parent_config = deep_merge(config, parent_config)
        return parent_config
    return config


def load_segmentation_task_config(yaml_filepath: str) -> SegmentationTaskConfig:
    return SegmentationTaskConfig.model_validate(_load_config_to_dict(yaml_filepath))


def load_recognition_task_config(yaml_filepath: str) -> RecognitionTaskConfig:
    return RecognitionTaskConfig.model_validate(_load_config_to_dict(yaml_filepath))

def load_contrastive_recognition_task_config(yaml_filepath: str) -> ContrastiveRecognitionTask:
    return ContrastiveRecognitionTask.model_validate(_load_config_to_dict(yaml_filepath))
