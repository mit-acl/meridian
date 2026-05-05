import dataclasses
import logging
import os
import subprocess

import numpy as np
from typing import Tuple, List, Union
from os.path import expandvars, expanduser
import open3d as o3d
import yaml

logger = logging.getLogger(__name__)


def _yaml_represent_numpy(dumper, data):
    if isinstance(data, np.floating):
        if np.isinf(data):
            return dumper.represent_scalar(
                "tag:yaml.org,2002:float", ".inf" if data > 0 else "-.inf"
            )
        return dumper.represent_float(float(data))
    if isinstance(data, np.integer):
        return dumper.represent_int(int(data))
    if isinstance(data, np.ndarray):
        return dumper.represent_list(data.tolist())
    return dumper.represent_data(data)


def _get_yaml_dumper():
    dumper = yaml.Dumper
    dumper.add_representer(np.float64, _yaml_represent_numpy)
    dumper.add_representer(np.float32, _yaml_represent_numpy)
    dumper.add_representer(np.int64, _yaml_represent_numpy)
    dumper.add_representer(np.int32, _yaml_represent_numpy)
    dumper.add_representer(np.ndarray, _yaml_represent_numpy)
    dumper.add_representer(tuple, lambda d, data: d.represent_list(list(data)))
    return dumper


def save_params(output_dir, *param_objects):
    """Save parameter objects to params.yaml in the output directory.

    Merges with existing params.yaml if present, so multiple pipeline stages
    can append their params to the same file.
    """
    params_path = os.path.join(output_dir, "params.yaml")

    # Load existing params if file exists
    existing = {}
    if os.path.isfile(params_path):
        with open(params_path, "r") as f:
            existing = yaml.full_load(f) or {}

    # Build dict from param objects
    for obj in param_objects:
        key = obj.params_key
        existing[key] = dataclasses.asdict(obj)

    with open(params_path, "w") as f:
        yaml.dump(
            existing,
            f,
            Dumper=_get_yaml_dumper(),
            default_flow_style=False,
            sort_keys=False,
        )


def save_commit_hash(output_dir):
    """Write the current git commit hash to commit.txt in the output directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            with open(os.path.join(output_dir, "commit.txt"), "w") as f:
                f.write(commit_hash + "\n")
        else:
            logger.warning("Could not get git commit hash: %s", result.stderr.strip())
    except FileNotFoundError:
        logger.warning("git not found — skipping commit.txt")


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


def expandvars_recursive(source):
    """Recursively expands environment variables in the given path."""
    if type(source) is str:
        path = source
        while True:
            expanded_path = expandvars(path)
            if expanded_path == path:
                return expanduser(expanded_path)
            path = expanded_path

    elif type(source) is list:
        ret_list = []
        for element in source:
            ret_list.append(expandvars_recursive(element))
        return ret_list

    elif type(source) is dict:
        ret_dict = {}
        for key, element in source.items():
            ret_dict[key] = expandvars_recursive(element)
        return ret_dict

    else:
        return source


def clean_up_points(
    points: np.ndarray,
    voxel_size: float = None,
    outlier_removal_std: float = None,
    dbscan_epsilon: float = None,
    dbscan_min_points: int = 10,
) -> np.ndarray:
    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if outlier_removal_std is not None:
            pcd, _ = pcd.remove_statistical_outlier(10, outlier_removal_std)

        if pcd.is_empty():
            points = None
        else:
            points = np.asarray(pcd.points)

    if points is not None and dbscan_epsilon is not None:
        # Perform DBSCAN clustering
        labels = np.array(
            pcd.cluster_dbscan(
                eps=dbscan_epsilon,
                min_points=dbscan_min_points,
            )
        )

        # Number of clusters, ignoring noise if present
        max_label = labels.max() if labels.size > 0 else -1

        if max_label < 0:
            # No clusters found (all points are noise)
            points = None
        else:
            # get largest cluster
            cluster_sizes = np.zeros(max_label + 1)
            for i in range(max_label + 1):
                cluster_sizes[i] = np.sum(labels == i)
            max_cluster = np.argmax(cluster_sizes)

            # Filter out any points not belonging to max cluster
            filtered_indices = np.where(labels == max_cluster)[0]
            points = points[filtered_indices]

    return points


def vstack_opt(arrs: List[Union[np.ndarray, None]]) -> Union[np.ndarray, None]:
    """Vertically stack arrays, handling None and empty arrays."""
    arrs = [arr for arr in arrs if arr is not None]
    if not arrs:
        return None

    nonzero_arrs = [arr for arr in arrs if arr.size > 0]
    if not nonzero_arrs:
        return arrs[0]

    return np.vstack(nonzero_arrs)
