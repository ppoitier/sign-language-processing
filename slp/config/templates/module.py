from pydantic import BaseModel


class ModuleConfig(BaseModel):
    module_name: str
    module_kwargs: dict = {}
