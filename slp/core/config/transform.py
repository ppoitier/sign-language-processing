from typing import Any

from pydantic import BaseModel, Field


class TransformConfig(BaseModel):
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
