"""Flatten 3D dense submaps to 2D and convert their segments to AerialSegments."""

from copy import deepcopy
from typing import List

from meridian.map3d.submap import Submap
from meridian.segment.aerial_segment import AerialSegment
from meridian.segment.segment_types import GeneralSegment
from meridian.utils import clean_up_points


def flatten_3d_submap(
    submap_3d: Submap,
    outlier_removal_std: float,
    dbscan_epsilon: float,
    dbscan_min_points: int,
) -> Submap:
    """Project a 3D dense submap onto z=0 and clean up each segment's points."""
    map_2d = deepcopy(submap_3d)
    to_rm = []
    for seg in map_2d.segments:
        seg.dense_points[:, 2] = 0.0
        if (
            getattr(seg, "occluded_points", None) is not None
            and len(seg.occluded_points) > 0
        ):
            seg.occluded_points[:, 2] = 0.0
        try:
            _cleanup_segment_points(
                seg,
                outlier_removal_std=outlier_removal_std,
                dbscan_epsilon=dbscan_epsilon,
                dbscan_min_points=dbscan_min_points,
            )
        except Exception:
            to_rm.append(seg)
            continue
        if seg.dense_points is None or len(seg.dense_points) < 2:
            to_rm.append(seg)
    for seg in to_rm:
        map_2d.segments.remove(seg)
    return map_2d


def submap_2d_to_aerial(submap_2d: Submap) -> List[AerialSegment]:
    """Convert each segment in a flattened submap to an AerialSegment."""
    aerial_segments = []
    for seg in submap_2d.segments:
        aerial_segment = AerialSegment(
            id=seg.id,
            center=seg.point.reshape(-1)[:2],
            area=None,
            points=seg.dense_points[:, :2],
            semantic_descriptor=seg.cos_feature,
            first_seen=seg.first_seen,
            last_seen=seg.last_seen,
        )
        aerial_segment.calculate_area_from_convex_hull()
        aerial_segments.append(aerial_segment)
    return aerial_segments


def _cleanup_segment_points(
    segment: GeneralSegment,
    outlier_removal_std: float,
    dbscan_epsilon: float,
    dbscan_min_points: int,
):
    segment.dense_points = clean_up_points(
        segment.dense_points,
        voxel_size=segment.voxel_size,
        outlier_removal_std=outlier_removal_std,
        dbscan_epsilon=dbscan_epsilon,
        dbscan_min_points=dbscan_min_points,
    )
