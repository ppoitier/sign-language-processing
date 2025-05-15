from pydantic import BaseModel


class ModuleConfig(BaseModel):
    module_name: str
    module_kwargs: dict = {}
    checkpoint_path: str | None = None
