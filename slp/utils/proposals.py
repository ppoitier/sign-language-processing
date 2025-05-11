import numpy as np


def generate_proposals(start_offsets, end_offsets, min_duration=1, return_indices=False):
    max_t = start_offsets.shape[0]
    # Generate all possible start and end times
    t = np.arange(max_t, dtype='float32')
    start_times = t - start_offsets
    end_times = t + end_offsets
    proposals = np.stack((start_times, end_times), axis=-1)
    # Filter invalid proposals
    valid_mask = (
            (proposals[:, 1] >= proposals[:, 0]) &
            (proposals[:, 0] >= 0) &
            (proposals[:, 1] < max_t) &
            ((proposals[:, 1] - proposals[:, 0] + 1) >= min_duration)
    )
    if return_indices:
        return proposals[valid_mask], t[valid_mask].astype('int32')
    return proposals[valid_mask]
