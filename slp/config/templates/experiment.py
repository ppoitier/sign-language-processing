from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    id: str
    suffix: str = 'default'
    seed: int | Literal['random'] = 'random'
    task_datetime: datetime = Field(default_factory=datetime.now)
    output_dir: str
    mlflow_uri: str |None = None
    show_progress_bar: bool = False
    debug: bool = False
