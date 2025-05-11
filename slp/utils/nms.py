import numpy as np


def compute_iou(proposal: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    """
    Compute IoU between one proposal and an array of proposals.

    Args:
    proposal (np.ndarray): A single proposal of shape (2,) containing [start, end].
    Proposals (np.ndarray): An array of proposals with shape (N, 2) where each row is [start, end].

    Returns:
    np.ndarray: An array of IoU values with shape (N,).
    """
    intersections = np.maximum(
        0,
        np.minimum(proposal[1], proposals[:, 1]) - np.maximum(proposal[0], proposals[:, 0]),
    )
    proposal_duration = proposal[1] - proposal[0]
    proposals_durations = proposals[:, 1] - proposals[:, 0]
    unions = proposal_duration + proposals_durations - intersections
    return np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions != 0)


def soft_nms(proposals: np.ndarray, scores: np.ndarray, method='gaussian', sigma=0.1, threshold=0.2):
    # N = proposals.shape[0]
    for i in range(proposals.shape[0]):
        max_score_idx = i + scores[i:].argmax()
        proposals[i], proposals[max_score_idx] = proposals[max_score_idx].copy(), proposals[i].copy()
        scores[i], scores[max_score_idx] = scores[max_score_idx], scores[i]
        ious = compute_iou(proposals[i], proposals[i+1:])
        if method == 'linear':
            weights = 1 - ious
        elif method == 'gaussian':
            weights = np.exp(-(ious * ious) / sigma)
        else:
            raise ValueError("Method must be either 'linear' or 'gaussian'")
        scores[i+1:] *= weights
    keep = scores > threshold
    return proposals[keep], scores[keep]
