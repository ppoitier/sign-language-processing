import random

import numpy as np


def compute_window_indices(
    sequence_length: int, window_size: int, stride: int
) -> np.ndarray:
    """
    Compute the start and end indices for each window, including a possibly shorter last window.

    Args:
        sequence_length: Total length of the sequence
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        np.ndarray: Array of shape (n_windows, 2) containing start and end indices for each window
    """
    start_indices = np.arange(0, sequence_length, stride)
    end_indices = np.minimum(start_indices + window_size, sequence_length)
    valid_windows = start_indices != end_indices
    return np.column_stack((start_indices[valid_windows], end_indices[valid_windows]))


def get_segments_in_range(segments: np.ndarray, start: int, end: int) -> np.ndarray:
    return (
        np.clip(
            segments[(segments[:, 0] < end) & (segments[:, 1] >= start)],
            a_min=start,
            a_max=end,
        )
        - start
    )


def get_window_from_instance(
    instance: dict,
    start: int,
    end: int,
    add_metadata: bool = False,
    specials: dict | None = None,
    ignored: set[str] | None = None,
) -> dict:
    if isinstance(instance, np.ndarray) or isinstance(instance, list):
        return instance[start:end]
    elif isinstance(instance, dict):
        new_instance = dict()
        if add_metadata:
            new_instance["start"] = start
            new_instance["end"] = end
        for k, v in instance.items():
            if ignored is not None and k in ignored:
                continue
            if specials is not None and k in specials:
                new_instance[k] = specials[k](v, start, end)
            else:
                new_instance[k] = get_window_from_instance(v, start, end, ignored=ignored, specials=specials)
        return new_instance
    else:
        return instance


def get_windows_from_segmentation_instance(
    instance: dict, window_indices: np.ndarray
) -> list[dict]:
    return [
        get_window_from_instance(
            instance,
            start,
            end,
            add_metadata=True,
            ignored={"segment_classes"},
            specials={"segments": get_segments_in_range},
        )
        for start, end in window_indices
    ]


def convert_segmentation_instances_to_windows(
    instances: list[dict], window_size: int, stride: int
):
    new_instances = []
    for instance in instances:
        seq_len = instance["poses"]["upper_pose"].shape[0]
        window_indices = compute_window_indices(seq_len, window_size, stride)
        new_instances += get_windows_from_segmentation_instance(
            instance, window_indices
        )
    return new_instances


def filter_empty_segmentation_windows(instances: list[dict], max_empty_windows: int):
    empty_window_indices = [
        index
        for index, instance in enumerate(instances)
        if instance["segments"].shape[0] < 1
    ]
    kept_empty_windows = random.sample(empty_window_indices, max_empty_windows)
    removed_windows_indices = set(empty_window_indices).difference(
        set(kept_empty_windows)
    )
    return [
        instance
        for i, instance in enumerate(instances)
        if i not in removed_windows_indices
    ]
