import random
from typing import Literal

import numpy as np
import torch


def set_seed(seed: int | Literal['random']):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    if seed == 'random':
        seed = random.randint(a=0, b=1_000_000)
        print(f"Random seed: {seed:_}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)