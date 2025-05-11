import numpy as np


def bio_probabilities_to_segments(
    bio_probabilities: np.ndarray,
    threshold_b: float,
    threshold_o: float,
) -> np.ndarray:
    prob_b = bio_probabilities[:, 1]
    prob_o = bio_probabilities[:, 0]
    n_frames = len(prob_b)
    if n_frames == 0:
        return np.empty((0, 2), dtype=int)
    segments = []
    starting_phase = False
    active_segment_start = None
    for i in range(n_frames):
        if not starting_phase and prob_b[i] > threshold_b:
            starting_phase = True
            active_segment_start = i
        elif starting_phase and prob_b[i] <= threshold_b:
            starting_phase = False
        if (
            not starting_phase
            and (active_segment_start is not None)
            and (prob_b[i] > threshold_b or prob_o[i] > threshold_o)
        ):
            segments.append((active_segment_start, i - 1))
            active_segment_start = None
    # After the loop, if a segment is still active, it extends to the last frame
    if active_segment_start is not None:
        segments.append((active_segment_start, n_frames - 1))
    if not segments:
        return np.empty((0, 2), dtype=int)
    return np.array(segments, dtype=int)
