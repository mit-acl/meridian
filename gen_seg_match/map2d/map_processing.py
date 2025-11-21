import numpy as np
import copy
from typing import List

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint


def merge_lines(line1: SegmentLine, line2: SegmentLine) -> SegmentLine:
    """Merge two lines into one by taking the endpoints that are farthest apart."""
    len_line1 = np.linalg.norm(line1.endpoints[1] - line1.endpoints[0])
    len_line2 = np.linalg.norm(line2.endpoints[1] - line2.endpoints[0])
    direction1 = line1.get_direction() * len_line1 + line2.get_direction() * len_line2
    direction2 = line1.get_direction() * len_line1 - line2.get_direction() * len_line2
    direction = (
        direction1
        if np.linalg.norm(direction1) > np.linalg.norm(direction2)
        else direction2
    )
    direction /= np.linalg.norm(direction)
    # TODO: maybe weight the direction by the length?
    line1_infinite = SegmentLine(-1, line1.get_point(), line1.get_direction())
    line2_infinite = SegmentLine(-1, line2.get_point(), line2.get_direction())
    if line1.is_parallel_to(line2):
        point1 = line1.get_point()
        point2 = line2_infinite.closest_point_to_point(point1)
        point = (point1 + point2) / 2
    else:
        point1, point2 = line1_infinite.closest_points(line2_infinite)
        point = (point1 + point2) / 2

    merged_infinite = SegmentLine(-1, point, direction)

    # project endpoints of both lines onto the merged line
    endpoint_candidates = []
    for pt in [
        line1.endpoints[0],
        line1.endpoints[1],
        line2.endpoints[0],
        line2.endpoints[1],
    ]:
        endpoint_candidates.append(merged_infinite.closest_point_to_point(pt))

    pt1 = None
    pt2 = None
    max_dist = -1
    for i in range(len(endpoint_candidates)):
        for j in range(i + 1, len(endpoint_candidates)):
            dist = np.linalg.norm(endpoint_candidates[i] - endpoint_candidates[j])
            if dist > max_dist:
                max_dist = dist
                pt1 = endpoint_candidates[i]
                pt2 = endpoint_candidates[j]

    # TODO: we should probably keep track of the history of cosine features as we are
    # merging lines. Also, should probably weight by length.
    if line1.cos_feature is not None and line2.cos_feature is not None:
        merged_cos_feature = (line1.cos_feature + line2.cos_feature) / 2
        merged_cos_feature /= np.linalg.norm(merged_cos_feature)
    else:
        merged_cos_feature = None
    assert type(line1.first_seen) == type(line2.first_seen)
    assert type(line1.last_seen) == type(line2.last_seen)
    first_seen = line1.first_seen
    last_seen = line1.last_seen
    if line2.first_seen is not None and line2.first_seen < first_seen:
        first_seen = line2.first_seen
    if line2.last_seen is not None and line2.last_seen > last_seen:
        last_seen = line2.last_seen
    return SegmentLine.from_endpoints(
        -1,
        pt1,
        pt2,
        cos_feature=merged_cos_feature,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def merge_points(pt1, pt2):
    new_point = (pt1.get_point() + pt2.get_point()) / 2
    return SegmentPoint(-1, new_point)


def clean_up_line_map(
    lines: List[SegmentLine],
    max_iter: int = 1000,
    angle_tol: float = np.deg2rad(5),
    dist_tol: float = 0.5,
) -> List[SegmentLine]:
    def merge_check(line1, line2):
        return (
            line1.is_parallel_to(line2, tol=angle_tol)
            and line1.min_dist_to(line2) < dist_tol
        )

    return _clean_up_map(lines, merge_check, merge_lines, max_iter)


def clean_up_point_map(
    points: List[SegmentPoint], max_iter: int = 1000, dist_tol: float = 0.5
) -> List[SegmentPoint]:
    def merge_check(pt1, pt2):
        return np.linalg.norm(pt1.get_point() - pt2.get_point()) < dist_tol

    return _clean_up_map(points, merge_check, merge_points, max_iter)


def _clean_up_map(
    objects: list,
    merge_check: callable,
    merge_objects: callable,
    max_iter: int = 1000,
) -> list:
    objects = copy.deepcopy(objects)
    prev_objects = copy.deepcopy(objects)
    for outer_iter in range(max_iter):
        outer_changed = False
        for i in range(len(prev_objects) - 1, -1, -1):
            changed = False
            for j in range(i + 1, len(objects)):
                obj_i = prev_objects[i]
                obj_j = objects[j]
                # print(line_i.is_parallel_to(line_j, tol=angle_tol), line_i.min_dist_to(line_j))
                if merge_check(obj_i, obj_j):
                    objects[i] = merge_objects(obj_i, obj_j)
                    del objects[j]
                    changed = True
                    outer_changed = True
                    break
            if changed:
                continue
        if not outer_changed:
            break
        prev_objects = copy.deepcopy(objects)

    return objects, outer_iter + 1
