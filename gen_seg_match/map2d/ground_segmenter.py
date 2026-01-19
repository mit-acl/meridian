import numpy as np
from typing import List
from copy import deepcopy
import open3d as o3d

from roman.object.segment import Segment as ROMANSegment

from gen_seg_match.params import GroundSegmenterParams
from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.segment.segment_types import SegmentList, GeneralSegment
from gen_seg_match.map3d.submap import Submap
from gen_seg_match.utils import clean_up_points


class GroundSegmenter:
    def __init__(self, params: GroundSegmenterParams):
        self.params = params

    def flatten_3d_submap(self, submap_3d: Submap) -> Submap:
        map_2d = deepcopy(submap_3d)
        to_rm = []
        for seg in map_2d.segments:
            seg.dense_points[:, 2] = 0.0
            try:
                self.cleanup_points(seg)
            except Exception as e:
                to_rm.append(seg)
            if len(seg.dense_points) < 2:
                to_rm.append(seg)
        return map_2d

    def submap_2d_to_aerial(self, submap_2d: Submap) -> List[GeneralSegment]:
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

    def cleanup_points(self, segment: GeneralSegment):
        segment.dense_points = clean_up_points(
            segment.dense_points,
            voxel_size=segment.voxel_size,
            outlier_removal_std=self.params.outlier_removal_std,
            dbscan_epsilon=self.params.dbscan_epsilon,
            dbscan_min_points=self.params.dbscan_min_points,
        )
