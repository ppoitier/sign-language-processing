from abc import ABC, abstractmethod
from torch import Tensor


class SegmentDecoder(ABC):
    """Decodes model logits into a list of predicted segments.

    Each decoder takes a dict of logits (single instance, no batch dim)
    and returns an (S, 2) long tensor of [start_frame, end_frame] pairs,
    sorted by start time.
    """

    @abstractmethod
    def decode(self, logits: dict[str, Tensor], n_classes: int) -> Tensor:
        ...

    def decode_batch(
        self, logits: dict[str, Tensor], n_classes: int, batch_size: int
    ) -> list[Tensor]:
        """Decode each instance in a batch independently.

        Args:
            logits: dict of head_name -> tensor of shape (B, C, T) or (B, 2, T).
            n_classes: number of classes.
            batch_size: batch size B.

        Returns:
            List of (S_i, 2) tensors, one per instance.
        """
        results = []
        for idx in range(batch_size):
            instance_logits = {k: v[idx] for k, v in logits.items()}
            results.append(self.decode(instance_logits, n_classes))
        return results