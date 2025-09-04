from pydantic import BaseModel, Field


class SegmentCodecConfig(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
