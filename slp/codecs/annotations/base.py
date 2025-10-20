from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from torch import Tensor


class AnnotationCodec(ABC):
    @abstractmethod
    def encode(self, annotations: np.ndarray, n_frames: int) -> dict[str, np.ndarray]:
        raise NotImplementedError("Annotation encoding is not implemented.")

    def decode(self, logits: dict[str, Union[Tensor, np.ndarray]], n_classes: int) -> Tensor:
        raise NotImplementedError("Annotation decoding is not implemented.")

    def decode_batch(self, logits: dict[str, Tensor], n_classes: int, batch_size: int) -> list[Tensor]:
        return [self.decode({k: v[idx] for k, v in logits.items()}, n_classes) for idx in range(batch_size)]
