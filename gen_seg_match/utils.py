import numpy as np
from typing import Tuple, List, Union


def sort_time_intervals(
    intervals: Union[List[Tuple[float, float]], np.ndarray], ref_time: float
) -> np.ndarray:
    """
    Sort time intervals by distance to the reference time.

    Intervals that contain the reference time are ordered first, internally sorted by their average time's
    distance to the reference time. Remaining intervals are sorted by the minimum distance from their edges.

    Args:
        intervals (Union[np.ndarray, List[Tuple[float, float]]]): List of N time intervals represented as
            (start_time, end_time).
        ref_time (float): The reference time to sort against.
    Returns:
        np.ndarray: Indices that would sort the intervals as described, in (N,) integer numpy array.
    """

    intervals = np.array(intervals)

    contains_ref = (intervals[:, 0] <= ref_time) & (intervals[:, 1] >= ref_time)
    contains_ref_indices = np.nonzero(contains_ref)[0]
    not_contains_ref_indices = np.nonzero(~contains_ref)[0]

    contains_ref_avg_times = (
        intervals[contains_ref][:, 0] + intervals[contains_ref][:, 1]
    ) / 2
    contains_ref_distances = np.abs(contains_ref_avg_times - ref_time)

    not_contains_ref_distances = np.minimum(
        np.abs(intervals[not_contains_ref_indices][:, 0] - ref_time),
        np.abs(intervals[not_contains_ref_indices][:, 1] - ref_time),
    )

    contains_ref_indices = contains_ref_indices[np.argsort(contains_ref_distances)]
    not_contains_ref_indices = not_contains_ref_indices[
        np.argsort(not_contains_ref_distances)
    ]

    return np.concatenate([contains_ref_indices, not_contains_ref_indices])
