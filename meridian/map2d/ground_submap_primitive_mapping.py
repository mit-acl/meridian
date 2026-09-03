import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from meridian.map2d.segment_to_primitive import (
    SegmentToPrimitiveConverter,
    line_is_valid,
)
from meridian.map3d.dense3d_to_dense2d import (
    flatten_3d_submap,
    submap_2d_to_aerial,
)
from meridian.map3d.submap import FrameType, Submap
from meridian.params.ground_segmenter_params import GroundSegmenterParams
from meridian.params.segment_to_primitive_params import GroundSubmapParams
from meridian.primitive.primitive_list import PrimitiveList
from meridian.primitive.dense_segment import DenseSegment, get_roman_ratio_feature

from meridian.map3d.map import SegmentMap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroundSubmapIntermediates:
    flattened_submap: Submap = None
    aerial_segments: list = None
    general_segments: PrimitiveList = None


@dataclass
class GroundSegmentationResult:
    submaps: List[Submap]
    intermediates: Optional[List[GroundSubmapIntermediates]] = None


# ---------------------------------------------------------------------------
# GroundSubmapPrimitiveMapping
# ---------------------------------------------------------------------------


class GroundSubmapPrimitiveMapping:
    """Creates sparse 2D primitive submaps from 3D ground segment maps."""

    def __init__(
        self,
        submap_params: GroundSubmapParams,
        converter: SegmentToPrimitiveConverter,
        ground_segmenter_params: GroundSegmenterParams,
        place_recognition=None,
    ):
        self.submap_params = submap_params
        self.converter = converter
        self.ground_segmenter_params = ground_segmenter_params
        self.place_recognition = place_recognition

    # ------------------------------------------------------------------
    # 3D map -> dense submaps
    # ------------------------------------------------------------------

    def create_submaps_from_map(
        self,
        ground_map: SegmentMap,
    ) -> List[Submap]:
        """Create 3D dense submaps from a ground map by spatial/temporal windowing.

        Returns submaps in CAMERA frame with segments transformed to submap-local coords.
        """
        dense_segments = []
        for seg in ground_map.segments:
            ds = DenseSegment(
                id=seg.id,
                dense_points=seg.points,
                ratio_feature=get_roman_ratio_feature(seg),
                cos_feature=seg.semantic_descriptor,
                first_seen=seg.first_seen,
                last_seen=seg.last_seen,
                occluded_points=seg.occluded_points,
                history=getattr(seg, "history", []),
            )
            ds.voxel_size = getattr(seg, "voxel_size", None)
            dense_segments.append(ds)

        params = self.submap_params
        dist_m = params.ground_submap_dist_m
        rad_m = params.ground_submap_rad_m
        time_s = params.ground_submap_time_s

        # Sample poses along trajectory spaced ground_submap_dist_m apart
        sampled_indices = []
        last_position = None
        for i, pose in enumerate(ground_map.trajectory):
            position = pose[:3, 3]
            if (
                last_position is None
                or np.linalg.norm(position - last_position) >= dist_m
            ):
                sampled_indices.append(i)
                last_position = position

        _attach_frame_descriptors = (
            self.place_recognition is not None
            and self.place_recognition.comparison == "image"
        )
        if _attach_frame_descriptors:
            self.place_recognition.precompute_ground_map_data(ground_map)

        # For each sampled pose, collect segments with dense points within radius
        submaps = []
        for k, idx in enumerate(sampled_indices):
            pose = ground_map.trajectory[idx]
            center = pose[:3, 3]
            submap_segments = []
            submap_time = ground_map.times[idx]

            for seg in dense_segments:
                if seg.dense_points is None:
                    continue
                # Temporal overlap check
                if seg.first_seen is not None and seg.last_seen is not None:
                    if seg.first_seen > submap_time + time_s:
                        continue
                    if seg.last_seen < submap_time - time_s:
                        continue
                dists = np.linalg.norm(seg.dense_points - center, axis=1)
                mask = dists <= rad_m
                if not np.any(mask):
                    continue
                seg_copy = seg.copy()
                seg_copy.voxel_size = getattr(seg, "voxel_size", None)
                seg_copy.dense_points = seg.dense_points[mask].copy()
                seg_copy.point = np.mean(seg_copy.dense_points, axis=0)

                # Filter existing occluded points to within radius,
                # then remove any whose (x,y) grid cell is occupied by a
                # regular point
                existing_occ = getattr(seg, "occluded_points", None)
                if existing_occ is not None and len(existing_occ) > 0:
                    occ_dists = np.linalg.norm(existing_occ - center, axis=1)
                    existing_occ = existing_occ[occ_dists <= rad_m]
                if existing_occ is not None and len(existing_occ) > 0:
                    grid = params.occluded_grid_voxel_size_m
                    dp_xy = np.floor(seg_copy.dense_points[:, :2] / grid).astype(int)
                    regular_keys = set(map(tuple, dp_xy))
                    occ_xy = np.floor(existing_occ[:, :2] / grid).astype(int)
                    occ_keep = np.array(
                        [tuple(k_) not in regular_keys for k_ in occ_xy]
                    )
                    existing_occ = existing_occ[occ_keep] if np.any(occ_keep) else None

                # Mark border points near radius cutoff as occluded
                border_mask = mask & (dists > rad_m - params.occluded_radius_thresh_m)
                border_occluded = seg.dense_points[border_mask].copy()

                all_occluded = [border_occluded]
                if existing_occ is not None and len(existing_occ) > 0:
                    all_occluded.append(existing_occ)
                total = sum(len(a) for a in all_occluded)
                seg_copy.occluded_points = (
                    np.concatenate(all_occluded, axis=0) if total > 0 else None
                )

                if (
                    len(seg_copy.dense_points)
                    >= self.converter.params.segment_min_points
                ):
                    submap_segments.append(seg_copy)

            if len(submap_segments) == 0:
                continue

            submap_descriptor = None
            if _attach_frame_descriptors:
                submap_descriptor = self.place_recognition.ground_descriptor(
                    None,
                    submap_segments=submap_segments,
                    center=center,
                    max_dist_m=rad_m,
                )

            submap = Submap(
                id=k,
                time=ground_map.times[idx],
                segments=PrimitiveList(submap_segments),
                pose=pose,
                segment_frame=FrameType.CAMERA,
                descriptor=submap_descriptor,
            )
            if self.place_recognition is not None:
                self.place_recognition.tag(submap)

            # Transform segments from odom frame to submap-local frame
            T_submap_odom = np.linalg.inv(submap.pose)
            for seg in submap.segments:
                seg.transform(T_submap_odom)

            submaps.append(submap)

        return submaps

    # ------------------------------------------------------------------
    # Single submap dense -> sparse 2D
    # ------------------------------------------------------------------

    def convert_submap_to_sparse_2d(
        self,
        submap: Submap,
        return_intermediates: bool = False,
    ) -> tuple:
        """Convert a single 3D dense submap to a sparse 2D submap.

        Args:
            submap: A 3D submap in CAMERA frame.
            return_intermediates: Whether to return intermediate results.

        Returns:
            (submap_2d, intermediate) tuple. intermediate is None if not requested.
        """
        assert submap.segment_frame == FrameType.CAMERA, (
            f"Expected submap segments in CAMERA frame, but got {submap.segment_frame}"
        )
        submap.segments.transform(submap.pose)

        gs_params = self.ground_segmenter_params
        flattened_submap = flatten_3d_submap(
            submap,
            outlier_removal_std=gs_params.outlier_removal_std,
            dbscan_epsilon=gs_params.dbscan_epsilon,
            dbscan_min_points=gs_params.dbscan_min_points,
        )
        aerial_segments = submap_2d_to_aerial(flattened_submap)
        aerial_segments = [
            seg
            for seg in aerial_segments
            if seg.get_alpha_shape(
                alpha=self.converter.params.alpha_shape_alpha,
                grid_downsample=self.converter.params.alpha_shape_grid_downsample,
                max_n_pts=self.converter.params.alpha_shape_max_n_pts,
                alpha_ref_size=self.converter.params.alpha_shape_ref_size_m,
            )
            is not None
        ]

        # Convert to sparse primitives (parallelized per-segment internally)
        general_segments = self.converter.convert(aerial_segments)

        sparse_general_segments = general_segments

        # Remove lines that are FOV border artifacts
        params = self.submap_params
        valid_lines = PrimitiveList()
        for line in sparse_general_segments.get_lines():
            parent_id = line.history[0] if line.history else None
            parent_seg = (
                flattened_submap.segments.get_segment_from_id(parent_id)
                if parent_id is not None
                else None
            )
            if parent_seg is None or line_is_valid(
                line,
                parent_seg,
                line_occlusion_num_samples=params.line_occlusion_num_samples,
                line_pt_dist_check_m=params.line_pt_dist_check_m,
                line_frac_near_points=params.line_frac_near_points,
                line_occlusion_req_non_occluded=params.line_occlusion_req_non_occluded,
            ):
                valid_lines.append(line)
        sparse_general_segments = sparse_general_segments.get_points() + valid_lines

        sparse_general_segments.reindex()
        for seg in sparse_general_segments:
            history_heights = [
                submap.segments.get_segment_from_id(id_hist).point.item(2)
                for id_hist in seg.history
            ]
            seg.height = np.mean(history_heights)

        # Compute place recognition descriptor
        ground_descriptor = submap.descriptor
        if (
            self.place_recognition is not None
            and self.place_recognition.comparison == "semantic-point-line"
        ):
            ground_descriptor = self.place_recognition.ground_descriptor(
                None, submap_segments=sparse_general_segments
            )

        submap_2d = Submap(
            id=submap.id,
            time=submap.time,
            segments=sparse_general_segments,
            pose=np.eye(4),
            segment_frame=FrameType.ODOMETRY,
            descriptor=ground_descriptor,
            metadata={"camera_pose": submap.pose},
        )
        if self.place_recognition is not None:
            self.place_recognition.tag(submap_2d)

        intermediate = None
        if return_intermediates:
            intermediate = GroundSubmapIntermediates(
                flattened_submap=flattened_submap,
                aerial_segments=aerial_segments,
                general_segments=general_segments,
            )

        return submap_2d, intermediate

    # ------------------------------------------------------------------
    # Batch conversion
    # ------------------------------------------------------------------

    def batch_convert(
        self,
        submaps: List[Submap],
        return_intermediates: bool = False,
        show_progress: bool = False,
    ) -> GroundSegmentationResult:
        """Convert a batch of 3D dense submaps to sparse 2D submaps.

        Processes submaps sequentially; parallelism is within each submap
        (per-segment) via the SegmentToPrimitiveConverter.
        """
        result_submaps = []
        intermediates_list = [] if return_intermediates else None

        iterator = enumerate(submaps)
        if show_progress:
            iterator = tqdm(iterator, total=len(submaps), desc="Ground segmentation")
        for k, submap in iterator:
            submap_2d, intermediate = self.convert_submap_to_sparse_2d(
                submap, return_intermediates=return_intermediates
            )
            result_submaps.append(submap_2d)
            if return_intermediates:
                intermediates_list.append(intermediate)

        return GroundSegmentationResult(
            submaps=result_submaps, intermediates=intermediates_list
        )
