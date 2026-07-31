from typing import Optional

import torch
from torch import nn


def load_pretrained_state_dict(
    checkpoint_path: str,
    prefix: str = "encoder.model.",
) -> dict[str, torch.Tensor]:
    """Extract the encoder weights from a self-supervised checkpoint.

    ``RepresentationLearningTrainer`` wraps the model in an ``SSLEncoder``, so
    its checkpoint stores keys like ``encoder.model.backbone.…`` whereas the
    supervised trainers expect ``backbone.…``. This strips the wrapper prefix
    so the weights can be loaded straight into a freshly built model.

    Args:
        checkpoint_path: a Lightning checkpoint written during pretraining.
        prefix: the wrapper prefix to strip.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str,
    prefix: str = "encoder.model.",
    freeze: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Initialise a downstream model from a pretrained encoder.

    Loading is non-strict on purpose: the downstream model has task heads that
    pretraining never saw, and it usually drops the projection head. The
    mismatches are printed so a silently untransferred backbone is impossible
    to miss.

    Args:
        model: the freshly built downstream model, modified in place.
        checkpoint_path: the self-supervised checkpoint.
        prefix: the wrapper prefix to strip, see ``load_pretrained_state_dict``.
        freeze: freeze the backbone, for linear-probe evaluation. Remember to
            keep it in eval mode too if it contains BatchNorm.
        verbose: report which keys were missing or unexpected.
    """
    state_dict = load_pretrained_state_dict(checkpoint_path, prefix=prefix)
    if not state_dict:
        raise ValueError(
            f"No parameter found under prefix '{prefix}' in {checkpoint_path}. "
            f"Is this a representation learning checkpoint?"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)

    if verbose:
        print(f"Loaded {len(state_dict)} pretrained tensors from {checkpoint_path}")
        if incompatible.missing_keys:
            print(f"  Randomly initialised (not in checkpoint): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Ignored (not in downstream model): {incompatible.unexpected_keys}")

    if freeze:
        backbone: Optional[nn.Module] = getattr(model, "backbone", None)
        if backbone is None:
            raise AttributeError(
                "Cannot freeze: the model has no 'backbone' attribute."
            )
        backbone.requires_grad_(False)
        backbone.eval()
        if verbose:
            print("  Backbone frozen.")

    return model
