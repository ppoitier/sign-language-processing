from pathlib import Path

import torch


def save_logits(logits, logits_dir: str | Path):
    logits_dir = Path(logits_dir)
    logits_dir.mkdir(parents=True, exist_ok=True)
    torch.save(logits, logits_dir / "logits.pt")

