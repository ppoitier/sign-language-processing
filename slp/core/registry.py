from typing import Callable


class Registry:
    """A simple registry to map string names to classes."""

    def __init__(self, registry_name: str):
        self.name = registry_name
        self._obj_map: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """Decorator to register a class with a specific name."""

        def wrap(cls: Callable) -> Callable:
            if name in self._obj_map:
                raise ValueError(
                    f"An object named '{name}' was already registered in '{self.name}'."
                )
            self._obj_map[name] = cls
            return cls

        return wrap

    def get(self, name: str) -> Callable:
        """Retrieves the class by name."""
        if name not in self._obj_map:
            raise KeyError(
                f"'{name}' is not found in the '{self.name}' registry. "
                f"Available options are: {list(self._obj_map.keys())}"
            )
        return self._obj_map[name]


BACKBONE_REGISTRY = Registry("backbone")
NECK_REGISTRY = Registry("neck")
HEAD_REGISTRY = Registry("head")

CRITERION_REGISTRY = Registry("criterion")

POSE_TRANSFORM_REGISTRY = Registry("pose-transform")
VIDEO_TRANSFORM_REGISTRY = Registry("video-transform")
