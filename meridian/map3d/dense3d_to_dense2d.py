"""Flatten 3D dense submaps to 2D and convert their segments to Segment2D objects."""

from copy import copy as _shallow_copy
from typing import List

from meridian.map3d.submap import Submap
from meridian.map2d.segment2d import Segment2D
from meridian.primitive.primitive import Primitive
from meridian.primitive.primitive_list import PrimitiveList
from meridian.utils import clean_up_points


def flatten_3d_submap(
    submap_3d: Submap,
    outlier_removal_std: float,
    dbscan_epsilon: float,
    dbscan_min_points: int,
) -> Submap:
    """Project a 3D dense submap onto z=0 and clean up each segment's points."""
    # Shallow-copy the submap and each segment, deep-copying only the point
    # arrays we actually mutate (z->0, then cleanup reassigns dense_points). This
    # replaces a full deepcopy of the object graph (large cos_feature descriptors,
    # histories, cached point clouds) — which dominated this stage — while leaving
    # submap_3d untouched for the later per-primitive height lookups.
    map_2d = _shallow_copy(submap_3d)
    new_segments = []
    for seg in submap_3d.segments:
        s = _shallow_copy(seg)
        # Invalidate point-derived caches so they can't be stale after we edit
        # the points on the copy.
        for _cache_attr in ("_pcd", "_gaussian", "_eigvals"):
            if hasattr(s, _cache_attr):
                setattr(s, _cache_attr, None)
        s.dense_points = seg.dense_points.copy()
        s.dense_points[:, 2] = 0.0
        if (
            getattr(seg, "occluded_points", None) is not None
            and len(seg.occluded_points) > 0
        ):
            s.occluded_points = seg.occluded_points.copy()
            s.occluded_points[:, 2] = 0.0
        try:
            _cleanup_segment_points(
                s,
                outlier_removal_std=outlier_removal_std,
                dbscan_epsilon=dbscan_epsilon,
                dbscan_min_points=dbscan_min_points,
            )
        except Exception:
            continue
        if s.dense_points is None or len(s.dense_points) < 2:
            continue
        new_segments.append(s)
    map_2d.segments = PrimitiveList(new_segments)
    return map_2d


def submap_2d_to_aerial(submap_2d: Submap) -> List[Segment2D]:
    """Convert each segment in a flattened submap to an Segment2D."""
    aerial_segments = []
    for seg in submap_2d.segments:
        aerial_segment = Segment2D(
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
    segment: Primitive,
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
