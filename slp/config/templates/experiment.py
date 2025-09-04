from pydantic import BaseModel, Field
from datetime import datetime


class ExperimentConfig(BaseModel):
    id: str
    suffix: str = 'default'
    seed: int = 42
    task_datetime: datetime = Field(default_factory=datetime.now)
    output_dir: str
    show_progress_bar: bool = False
    debug: bool = False
