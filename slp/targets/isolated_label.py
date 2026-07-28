from typing import Any

import torch
from sldl.targets.target import TargetEncoder


class IsolatedLabelTarget(TargetEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, sample: dict) -> Any:
        return torch.tensor(sample['label_id']).long()

    def collate(self, batch_targets: list[Any]) -> Any:
        return torch.stack(batch_targets).contiguous()
