import torch
import numpy as np


def to_numpy_array(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise ValueError(f"Unsupported type: {type(array)}")


def to_torch_tensor(array):
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise ValueError(f"Unsupported type: {type(array)}")


def to_array_type(array, arr_type):
    if arr_type == torch.Tensor:
        return to_torch_tensor(array)
    elif arr_type == np.ndarray:
        return to_numpy_array(array)
    else:
        raise ValueError(f"Unsupported type: {arr_type}")
