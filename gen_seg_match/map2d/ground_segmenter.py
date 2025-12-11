import numpy as np
from typing import List
from copy import deepcopy
import open3d as o3d

from roman.object.segment import Segment as ROMANSegment

from gen_seg_match.params import GroundSegmenterParams
from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.segment.segment_types import SegmentList, GeneralSegment
from gen_seg_match.map3d.submap import Submap


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
        if segment.dense_points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(segment.dense_points)
            pcd_sampled = pcd.voxel_down_sample(voxel_size=segment.voxel_size)
            if self.params.outlier_removal_std is not None:
                pcd_pruned, _ = pcd_sampled.remove_statistical_outlier(
                    10, self.params.outlier_removal_std
                )
            else:
                pcd_pruned = pcd_sampled

            if pcd_pruned.is_empty():
                segment.dense_points = None
            else:
                segment.dense_points = np.asarray(pcd_pruned.points)

        if segment.dense_points is not None:
            # Perform DBSCAN clustering
            labels = np.array(
                segment.pcd.cluster_dbscan(
                    eps=self.params.dbscan_epsilon,
                    min_points=self.params.dbscan_min_points,
                )
            )

            # Number of clusters, ignoring noise if present
            max_label = labels.max()

            # get largest cluster
            cluster_sizes = np.zeros(max_label + 1)
            for i in range(max_label + 1):
                cluster_sizes[i] = np.sum(labels == i)
            max_cluster = np.argmax(cluster_sizes)

            # Filter out any points not belonging to max cluster
            filtered_indices = np.where(labels == max_cluster)[0]
            segment.dense_points = segment.dense_points[filtered_indices]
