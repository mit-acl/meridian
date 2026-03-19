import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import circle_fit
from tqdm import tqdm

from gen_seg_match.cross_view.place_recognition import CrossViewPlaceRecognition
from gen_seg_match.map2d.aerial_segmenter import AerialSegmenter
from gen_seg_match.map2d.ground_segmenter import GroundSegmenter
from gen_seg_match.map2d.map_processing import clean_up_line_map, split_long_lines
from gen_seg_match.map3d.dense_to_sparse_converter import DenseToSparseConverter
from gen_seg_match.map3d.submap import FrameType, Submap
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.params import (
    CrossViewMatchingParams,
    GroundSegmenterParams,
    SubmapParams,
)
from gen_seg_match.pipeline.result import (
    PoseEstimationResult,
    PoseEstimationResultMatrix,
)
from gen_seg_match.register.registerer import (
    InsufficientAssociationsException,
    Registerer,
)
from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.segment.segment_types import (
    DenseSegment,
    SegmentLine,
    SegmentList,
    SegmentPoint,
)

from roman.map.map import ROMANMap
from gen_seg_match.map3d.map import SegmentMap

logger = logging.getLogger(__name__)

Crop = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AerialPatchIntermediates:
    patch_img: np.ndarray = None
    aerial_segments: List[AerialSegment] = None
    general_segments: SegmentList = None  # before sparsification


@dataclass
class AerialSegmentationResult:
    submaps: Dict[Crop, Submap]
    intermediates: Optional[Dict[Crop, AerialPatchIntermediates]] = None


@dataclass
class GroundSubmapIntermediates:
    flattened_submap: Submap = None
    aerial_segments: List[AerialSegment] = None
    general_segments: SegmentList = None


@dataclass
class GroundSegmentationResult:
    submaps: List[Submap]
    intermediates: Optional[List[GroundSubmapIntermediates]] = None


@dataclass
class SingleMatchResult:
    pose_result: PoseEstimationResult
    aerial_segs_processed: SegmentList = None
    ground_segs_processed: SegmentList = None
    matched_ground: SegmentList = None
    matched_aerial: SegmentList = None


@dataclass
class CrossViewMatchResult:
    results: Dict[str, PoseEstimationResultMatrix]  # ground_key -> matrix
    match_details: Dict[str, Dict[str, SingleMatchResult]] = field(
        default_factory=dict
    )  # ground_key -> aerial_key -> detail


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _convert_long_lines_to_infinite(segments: SegmentList, threshold: float):
    """Convert lines longer than threshold to infinite lines (no endpoints)."""
    if threshold is None:
        return
    for seg in segments.get_lines():
        if seg.get_length() > threshold:
            seg.endpoints = (None, None)


def _aerial_key_to_tuple(key: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in key.split("_"))


def _aerial_segments_to_general_segments_impl(
    segments: List[AerialSegment],
    pipeline_params: CrossViewMatchingParams,
    pixel_len_m: float = None,
    crop: Crop = None,
) -> SegmentList:
    """Convert aerial segments to general (point/line) segments.

    Module-level function so it can be called from multiprocessing workers.
    """
    if crop is not None:
        x1, y1, x2, y2 = crop
        min_dist_m = pipeline_params.aerial_min_dist_to_border_m
        x1_border = x1 * pixel_len_m + min_dist_m
        y1_border = y1 * pixel_len_m + min_dist_m
        x2_border = x2 * pixel_len_m - min_dist_m
        y2_border = y2 * pixel_len_m - min_dist_m
    else:
        x1_border = -np.inf
        y1_border = -np.inf
        x2_border = np.inf
        y2_border = np.inf

    def pt_within_border(pt):
        return x1_border <= pt[0] <= x2_border and y1_border <= pt[1] <= y2_border

    lines = []
    center_points = []
    for j, segment in enumerate(segments):
        if segment.area < pipeline_params.min_area_m_sq:
            continue
        if (
            segment.area < pipeline_params.point_max_area_m_sq
            and segment.max_extent < pipeline_params.point_max_len_m
        ):
            if not pt_within_border(segment.center):
                continue
            center_points.append(
                SegmentPoint(
                    j,
                    segment.center,
                    cos_feature=segment.semantic_descriptor,
                    first_seen=segment.first_seen,
                    last_seen=segment.last_seen,
                    history=[segment.id],
                )
            )
            continue
        alpha_shape = segment.get_alpha_shape(
            grid_downsample=pipeline_params.alpha_shape_grid_downsample,
            alpha=pipeline_params.alpha_shape_alpha,
            max_n_pts=pipeline_params.alpha_shape_max_n_pts,
            alpha_ref_size=pipeline_params.alpha_shape_ref_size_m,
        )
        if alpha_shape is None:
            continue

        if segment.area < pipeline_params.circle_point_max_area:
            alpha_pts = alpha_shape
            if alpha_pts.size == 0:
                continue
            if np.allclose(alpha_pts[0], alpha_pts[-1]):
                alpha_pts = alpha_pts[:-1]
            if len(alpha_pts) >= 3:
                xc, yc, r, s = circle_fit.least_squares_circle(alpha_pts)
                if (
                    r > 0
                    and s < pipeline_params.circle_point_rad_frac_fit_err * r
                    and r < pipeline_params.circle_point_max_rad
                ):
                    center = np.array([xc, yc])
                    if pt_within_border(center):
                        center_points.append(
                            SegmentPoint(
                                j,
                                center,
                                cos_feature=segment.semantic_descriptor,
                                first_seen=segment.first_seen,
                                last_seen=segment.last_seen,
                                history=[segment.id],
                            )
                        )
                    continue

        for i, pt0 in enumerate(alpha_shape):
            pt1 = alpha_shape[i + 1 if i + 1 < len(alpha_shape) else 0]
            keep = np.linalg.norm(pt1 - pt0) > pipeline_params.line_min_length_m
            keep &= pt_within_border(pt0) and pt_within_border(pt1)
            if keep:
                line = SegmentLine.from_endpoints(
                    j,
                    pt0,
                    pt1,
                    cos_feature=segment.semantic_descriptor,
                    first_seen=segment.first_seen,
                    last_seen=segment.last_seen,
                    history=[segment.id],
                )
                lines.append(line)
    result = SegmentList(center_points + lines)
    result.reindex()
    return result


def _line_is_valid_impl(
    line: SegmentLine,
    original_segment,
    pipeline_params: CrossViewMatchingParams,
) -> bool:
    """Check that a line is not just a FOV border artifact.

    Module-level function so it can be called from multiprocessing workers.
    """
    pt0, pt1 = line.endpoints
    if pt0 is None or pt1 is None:
        return True

    n = pipeline_params.line_occlusion_num_samples
    dist_thresh = pipeline_params.line_pt_dist_check_m

    t = np.linspace(0, 1, n).reshape(-1, 1)
    samples = pt0 + t * (pt1 - pt0)

    dense_pts = original_segment.dense_points
    occ_pts = getattr(original_segment, "occluded_points", None)

    if dense_pts is not None and len(dense_pts) > 0:
        dists_to_dense = np.linalg.norm(
            samples[:, None, :2] - dense_pts[None, :, :2], axis=2
        ).min(axis=1)
        frac_near = np.mean(dists_to_dense < dist_thresh)
        if frac_near < pipeline_params.line_frac_near_points:
            return False

    if occ_pts is not None and len(occ_pts) > 0:
        dists_to_occ = np.linalg.norm(
            samples[:, None, :2] - occ_pts[None, :, :2], axis=2
        ).min(axis=1)
        frac_non_occluded = np.mean(dists_to_occ >= dist_thresh)
        if frac_non_occluded < pipeline_params.line_occlusion_req_non_occluded:
            return False

    return True


def _process_aerial_patch_worker(
    pipeline_params,
    pixel_len_m,
    aerial_segments,
    crop,
    i,
    j,
    pose_flu,
    patch_size_m,
):
    """Post-process a single aerial patch (process-safe, CPU-only).

    Top-level function for use with ProcessPoolExecutor.
    """
    for segment in aerial_segments:
        segment.get_alpha_shape(
            alpha=pipeline_params.alpha_shape_alpha,
            grid_downsample=pipeline_params.alpha_shape_grid_downsample,
            max_n_pts=pipeline_params.alpha_shape_max_n_pts,
            alpha_ref_size=pipeline_params.alpha_shape_ref_size_m,
        )
    general_segments = _aerial_segments_to_general_segments_impl(
        aerial_segments,
        pipeline_params,
        pixel_len_m=pixel_len_m,
        crop=crop,
    )
    sparse_general_segments = (
        general_segments.get_points()
        + clean_up_line_map(
            general_segments.get_lines(),
            angle_tol=pipeline_params.line_merge_ang_thresh_rad,
            dist_tol=pipeline_params.line_merge_dist_thresh_m,
            perp_dist_tol=pipeline_params.line_merge_perp_dist_thresh_m,
            short_line_thresh=pipeline_params.line_merge_short_thresh_m,
            semantic_sim_thresh=pipeline_params.line_merge_semantic_sim,
        )[0]
    )
    _convert_long_lines_to_infinite(
        sparse_general_segments,
        pipeline_params.line_len_to_infinite,
    )
    sparse_general_segments.reindex()

    submap = Submap(
        id=(i, j),
        time=0.0,
        segments=sparse_general_segments,
        pose=pose_flu,
        segment_frame=FrameType.UTM,
        descriptor=None,
        metadata={
            "crop_center_m": np.array(
                [(i + 0.5) * patch_size_m, -(j + 0.5) * patch_size_m]
            )
        },
    )
    return submap, general_segments


def _process_ground_submap_worker(
    ground_segmenter,
    pipeline_params,
    place_recognition,
    k,
    submap,
    return_intermediates,
):
    """Process a single ground submap (process-safe, CPU-only).

    Top-level function for use with ProcessPoolExecutor.
    """
    assert submap.segment_frame == FrameType.CAMERA, (
        f"Expected submap segments in CAMERA frame, but got {submap.segment_frame}"
    )
    submap.segments.transform(submap.pose)

    flattened_submap = ground_segmenter.flatten_3d_submap(submap)
    aerial_segments = ground_segmenter.submap_2d_to_aerial(flattened_submap)
    aerial_segments = [
        seg
        for seg in aerial_segments
        if seg.get_alpha_shape(
            alpha=pipeline_params.alpha_shape_alpha,
            grid_downsample=pipeline_params.alpha_shape_grid_downsample,
            max_n_pts=pipeline_params.alpha_shape_max_n_pts,
            alpha_ref_size=pipeline_params.alpha_shape_ref_size_m,
        )
        is not None
    ]
    general_segments = _aerial_segments_to_general_segments_impl(
        aerial_segments, pipeline_params
    )
    sparse_general_segments = (
        general_segments.get_points()
        + clean_up_line_map(
            general_segments.get_lines(),
            angle_tol=pipeline_params.line_merge_ang_thresh_rad,
            dist_tol=pipeline_params.line_merge_dist_thresh_m,
            perp_dist_tol=pipeline_params.line_merge_perp_dist_thresh_m,
            short_line_thresh=pipeline_params.line_merge_short_thresh_m,
            semantic_sim_thresh=pipeline_params.line_merge_semantic_sim,
        )[0]
    )
    # Remove lines that are FOV border artifacts
    valid_lines = SegmentList()
    for line in sparse_general_segments.get_lines():
        parent_id = line.history[0] if line.history else None
        parent_seg = (
            flattened_submap.segments.get_segment_from_id(parent_id)
            if parent_id is not None
            else None
        )
        if parent_seg is None or _line_is_valid_impl(line, parent_seg, pipeline_params):
            valid_lines.append(line)
    sparse_general_segments = sparse_general_segments.get_points() + valid_lines

    _convert_long_lines_to_infinite(
        sparse_general_segments,
        pipeline_params.line_len_to_infinite,
    )

    sparse_general_segments.reindex()
    for seg in sparse_general_segments:
        history_heights = [
            submap.segments.get_segment_from_id(id_hist).point.item(2)
            for id_hist in seg.history
        ]
        seg.height = np.mean(history_heights)
    # For semantic-point-line, compute descriptor from point/line segments
    ground_descriptor = submap.descriptor
    if (
        place_recognition is not None
        and place_recognition.method == "semantic-point-line"
    ):
        ground_descriptor = place_recognition.ground_descriptor(
            None, submap_segments=sparse_general_segments
        )

    submap_2d = Submap(
        id=k,
        time=submap.time,
        segments=sparse_general_segments,
        pose=np.eye(4),
        segment_frame=FrameType.ODOMETRY,
        descriptor=ground_descriptor,
        metadata={"camera_pose": submap.pose},
    )

    intermediate = None
    if return_intermediates:
        intermediate = GroundSubmapIntermediates(
            flattened_submap=flattened_submap,
            aerial_segments=aerial_segments,
            general_segments=general_segments,
        )

    return submap_2d, intermediate


# ---------------------------------------------------------------------------
# CrossViewMatching — pure algorithm class (no I/O)
# ---------------------------------------------------------------------------


@dataclass
class CrossViewMatching:
    pipeline_params: CrossViewMatchingParams
    aerial_segmenter: AerialSegmenter
    matcher: SegmentMatcher
    registerer: Registerer
    ground_submap_params: SubmapParams = (
        None  # TODO This shouldn't be used anymore... Remove?
    )
    ground_segmenter: GroundSegmenter = None
    place_recognition: CrossViewPlaceRecognition = None

    def __post_init__(self):
        if self.ground_segmenter is None:
            self.ground_segmenter = GroundSegmenter(GroundSegmenterParams())
        if self.ground_submap_params is None:
            self.ground_submap_params = SubmapParams()

    # ------------------------------------------------------------------
    # Single-patch methods (unchanged from original)
    # ------------------------------------------------------------------

    def aerial_img_to_segments(self, img: np.ndarray, crop: Crop = None) -> SegmentList:
        segments = self.aerial_segmenter.run(img, crop=crop)
        thresh = self.pipeline_params.aerial_segment_line_rejection_thresh_m
        if thresh is not None:
            segments = [
                seg
                for seg in segments
                if not (
                    seg.obb_extents[0] < thresh[0] and seg.obb_extents[1] > thresh[1]
                )
            ]
        return segments

    def aerial_segments_to_general_segments(
        self, segments: List[AerialSegment], crop: Crop = None
    ) -> SegmentList:
        pixel_len_m = (
            self.aerial_segmenter.params.pixel_len_m if crop is not None else None
        )
        return _aerial_segments_to_general_segments_impl(
            segments,
            self.pipeline_params,
            pixel_len_m=pixel_len_m,
            crop=crop,
        )

    def ground_map_to_submaps(
        self,
        ground_map: Union[ROMANMap, SegmentMap],
    ) -> List[Submap]:
        dense_segments = []
        for seg in ground_map.segments:
            dense_segments.append(
                DenseSegment(
                    id=seg.id,
                    dense_points=seg.points,
                    ratio_feature=DenseToSparseConverter.get_roman_ratio_feature(seg),
                    cos_feature=seg.semantic_descriptor,
                    first_seen=seg.first_seen,
                    last_seen=seg.last_seen,
                    occluded_points=seg.occluded_points,
                    history=getattr(seg, "history", []),
                )
            )

        dist_m = self.pipeline_params.ground_submap_dist_m
        rad_m = self.pipeline_params.ground_submap_rad_m
        time_s = self.pipeline_params.ground_submap_time_s

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

        # Precompute ground map data for semantic-gem descriptors
        _attach_gem_descriptors = (
            self.place_recognition is not None
            and self.place_recognition.method in ("semantic-gem", "anyloc")
        )
        if _attach_gem_descriptors:
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
                # Check temporal overlap: segment must have been seen
                # within time_s of the submap time
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
                seg_copy.dense_points = seg.dense_points[mask].copy()
                seg_copy.point = np.mean(seg_copy.dense_points, axis=0)

                # Filter existing occluded points to within radius,
                # then remove any whose (x,y) grid cell is occupied by a
                # regular point — if observed at any z, the 2D projection
                # is not occluded
                existing_occ = getattr(seg, "occluded_points", None)
                if existing_occ is not None and len(existing_occ) > 0:
                    occ_dists = np.linalg.norm(existing_occ - center, axis=1)
                    existing_occ = existing_occ[occ_dists <= rad_m]
                if existing_occ is not None and len(existing_occ) > 0:
                    grid = self.pipeline_params.occluded_grid_voxel_size_m
                    dp_xy = np.floor(seg_copy.dense_points[:, :2] / grid).astype(int)
                    regular_keys = set(map(tuple, dp_xy))
                    occ_xy = np.floor(existing_occ[:, :2] / grid).astype(int)
                    occ_keep = np.array([tuple(k) not in regular_keys for k in occ_xy])
                    existing_occ = existing_occ[occ_keep] if np.any(occ_keep) else None

                # Mark border points near radius cutoff as occluded
                border_mask = mask & (
                    dists > rad_m - self.pipeline_params.occluded_radius_thresh_m
                )
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
                    >= self.pipeline_params.segment_min_points
                ):
                    submap_segments.append(seg_copy)

            if len(submap_segments) == 0:
                continue

            # Attach semantic-gem descriptors if available
            submap_descriptor = None
            if _attach_gem_descriptors:
                submap_descriptor = self.place_recognition.ground_descriptor(
                    None, submap_segments=submap_segments
                )

            submap = Submap(
                id=k,
                time=ground_map.times[idx],
                segments=SegmentList(submap_segments),
                pose=pose,
                segment_frame=FrameType.CAMERA,
                descriptor=submap_descriptor,
            )

            # Transform segments from odom frame to submap-local frame
            T_submap_odom = np.linalg.inv(submap.pose)
            for seg in submap.segments:
                seg.transform(T_submap_odom)

            submaps.append(submap)

        return submaps

    def _line_is_valid(self, line: SegmentLine, original_segment) -> bool:
        """Check that a line is not just a FOV border artifact."""
        return _line_is_valid_impl(line, original_segment, self.pipeline_params)

    # ------------------------------------------------------------------
    # Batch aerial segmentation (no I/O)
    # ------------------------------------------------------------------

    def _process_aerial_patch(
        self,
        aerial_segments,
        crop,
        i,
        j,
        pose_flu,
        patch_size_m,
    ):
        """Post-process a single aerial patch. Delegates to module-level worker."""
        pixel_len_m = (
            self.aerial_segmenter.params.pixel_len_m if crop is not None else None
        )
        return _process_aerial_patch_worker(
            self.pipeline_params,
            pixel_len_m,
            aerial_segments,
            crop,
            i,
            j,
            pose_flu,
            patch_size_m,
        )

    def batch_aerial_img_to_segments(
        self,
        img: np.ndarray,
        img_origin: np.ndarray = None,
        return_intermediates: bool = False,
        show_progress: bool = False,
    ) -> AerialSegmentationResult:
        h, w = img.shape[:2]
        px_per_m = 1.0 / self.aerial_segmenter.params.pixel_len_m

        patch_size_m = self.pipeline_params.aerial_img_patch_side_len_m
        patch_size_px = int(patch_size_m * px_per_m)

        overlap = self.pipeline_params.aerial_img_patch_overlap
        stride = int(patch_size_px * (1.0 - overlap))

        if stride <= 0:
            raise ValueError("Patch overlap too large; stride becomes non-positive.")

        logger.info(
            f"Aerial batch: img={w}x{h}px, pixel_len_m={self.aerial_segmenter.params.pixel_len_m:.6f}, "
            f"patch={patch_size_px}px ({patch_size_m}m), stride={stride}px, overlap={overlap}"
        )

        pose_flu = np.eye(4)
        pose_flu[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=float,
        )
        if img_origin is not None:
            pose_flu[0, 3] = img_origin[0]
            pose_flu[1, 3] = img_origin[1]

        submaps: Dict[Crop, Submap] = {}
        intermediates: Dict[Crop, AerialPatchIntermediates] = (
            {} if return_intermediates else None
        )

        patches = [
            (j, y1, i, x1)
            for j, y1 in enumerate(range(0, h - patch_size_px + 1, stride))
            for i, x1 in enumerate(range(0, w - patch_size_px + 1, stride))
        ]

        # Phase 1: serial segmentation (GPU-bound)
        segmentation_results = []
        seg_iterator = patches
        if show_progress:
            seg_iterator = tqdm(seg_iterator, desc="Aerial segmentation")
        for j, y1, i, x1 in seg_iterator:
            x2 = x1 + patch_size_px
            y2 = y1 + patch_size_px
            crop = (x1, y1, x2, y2)
            patch_img = img[y1:y2, x1:x2].copy()
            aerial_segments = self.aerial_img_to_segments(img, crop=crop)
            segmentation_results.append(
                (j, y1, i, x1, crop, patch_img, aerial_segments)
            )

        # Phase 2: parallel post-processing (CPU-bound, use processes to avoid GIL)
        max_workers = self.pipeline_params.sparse_conversion_max_threads
        pixel_len_m = self.aerial_segmenter.params.pixel_len_m
        pipeline_params = self.pipeline_params
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for j, y1, i, x1, crop, patch_img, aerial_segments in segmentation_results:
                future = executor.submit(
                    _process_aerial_patch_worker,
                    pipeline_params,
                    pixel_len_m,
                    aerial_segments,
                    crop,
                    i,
                    j,
                    pose_flu,
                    patch_size_m,
                )
                futures[future] = (j, y1, i, x1, crop, patch_img, aerial_segments)

            post_iterator = as_completed(futures)
            if show_progress:
                post_iterator = tqdm(
                    post_iterator,
                    total=len(futures),
                    desc="Aerial post-processing",
                )
            for future in post_iterator:
                j, y1, i, x1, crop, patch_img, aerial_segments = futures[future]
                submap, general_segments = future.result()

                # Place recognition descriptor (may use GPU — keep serial)
                if self.place_recognition is not None:
                    submap.descriptor = self.place_recognition.aerial_descriptor(
                        submap,
                        aerial_segmenter=self.aerial_segmenter,
                        img_bgr=img,
                        crop=crop,
                    )

                submaps[crop] = submap
                if return_intermediates:
                    intermediates[crop] = AerialPatchIntermediates(
                        patch_img=patch_img,
                        aerial_segments=aerial_segments,
                        general_segments=general_segments,
                    )

        return AerialSegmentationResult(submaps=submaps, intermediates=intermediates)

    # ------------------------------------------------------------------
    # Batch ground segmentation (no I/O)
    # ------------------------------------------------------------------

    def _process_ground_submap(self, k, submap, return_intermediates):
        """Process a single ground submap. Delegates to module-level worker."""
        return _process_ground_submap_worker(
            self.ground_segmenter,
            self.pipeline_params,
            self.place_recognition,
            k,
            submap,
            return_intermediates,
        )

    def batch_ground_to_sparse_2d_submaps(
        self,
        submaps: List[Submap],
        return_intermediates: bool = False,
        show_progress: bool = False,
    ) -> GroundSegmentationResult:
        max_workers = self.pipeline_params.sparse_conversion_max_threads
        ground_segmenter = self.ground_segmenter
        pipeline_params = self.pipeline_params
        place_recognition = self.place_recognition

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for k, submap in enumerate(submaps):
                future = executor.submit(
                    _process_ground_submap_worker,
                    ground_segmenter,
                    pipeline_params,
                    place_recognition,
                    k,
                    submap,
                    return_intermediates,
                )
                futures[future] = k

            result_submaps = [None] * len(submaps)
            intermediates_list = [None] * len(submaps) if return_intermediates else None

            post_iterator = as_completed(futures)
            if show_progress:
                post_iterator = tqdm(
                    post_iterator, total=len(futures), desc="Ground segmentation"
                )
            for future in post_iterator:
                k = futures[future]
                submap_2d, intermediate = future.result()
                result_submaps[k] = submap_2d
                if return_intermediates:
                    intermediates_list[k] = intermediate

        return GroundSegmentationResult(
            submaps=result_submaps, intermediates=intermediates_list
        )

    # ------------------------------------------------------------------
    # Cross-view matching (no I/O)
    # ------------------------------------------------------------------

    def cross_view_match(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        reference_trajectory=None,
        T_camera_flu: np.ndarray = None,
        matching_mode: str = None,
        translation_only: bool = None,
        show_progress: bool = False,
        local_to_pixel_fn=None,
    ) -> CrossViewMatchResult:
        """Match aerial and ground submaps without any I/O.

        Args:
            aerial_submaps: Aerial submap dict (key -> Submap).
            ground_submaps: Ground submap dict (key -> Submap).
            reference_trajectory: Any PoseData providing pose(time) -> 4x4.
                Used for rotation extraction and GT-based filtering.
            T_camera_flu: Camera-to-FLU transform.
            matching_mode: Override self.pipeline_params.matching_mode.
            translation_only: Override self.pipeline_params.translation_only.

        Returns:
            CrossViewMatchResult with per-ground PoseEstimationResultMatrix
            and per-pair SingleMatchResult.
        """
        mode = matching_mode or self.pipeline_params.matching_mode
        do_translation_only = (
            translation_only
            if translation_only is not None
            else self.pipeline_params.translation_only
        )

        aerial_submaps_2d, ground_submaps_2d = self._preprocess_submaps_2d(
            aerial_submaps, ground_submaps
        )

        aerial_key_to_tuple = _aerial_key_to_tuple
        aerial_x_max = max(aerial_key_to_tuple(key)[0] for key in aerial_submaps.keys())
        aerial_y_max = max(aerial_key_to_tuple(key)[1] for key in aerial_submaps.keys())

        # Precompute crop geometry
        pixel_len_m = self.aerial_segmenter.params.pixel_len_m
        px_per_m = 1.0 / pixel_len_m
        patch_size_px = int(self.pipeline_params.aerial_img_patch_side_len_m * px_per_m)
        stride = int(
            patch_size_px * (1.0 - self.pipeline_params.aerial_img_patch_overlap)
        )
        stride_m = stride * pixel_len_m
        patch_size_m_px = patch_size_px * pixel_len_m

        # Precompute VPR top-k patches if needed
        need_vpr = mode == "vpr" or (mode == "gt" and reference_trajectory is None)
        vpr_top_k_sets = {}
        if need_vpr:
            assert self.place_recognition is not None, (
                "VPR mode requires place_recognition to be configured"
            )
            sim_matrix, ground_keys_sorted, aerial_keys_sorted = (
                self.place_recognition.compute_similarity_matrix(
                    ground_submaps, aerial_submaps
                )
            )
            from gen_seg_match.pipeline.cross_view_place_recognition import (
                CrossViewPlaceRecognitionPipeline,
            )

            vpr_top_k = CrossViewPlaceRecognitionPipeline.compute_top_k_patches(
                sim_matrix,
                ground_keys_sorted,
                aerial_keys_sorted,
                self.place_recognition.params.k_nearest_neighbors,
            )
            vpr_top_k_sets = {gk: set(patches) for gk, patches in vpr_top_k.items()}

        all_results = {}
        all_details = {}

        iterator = ground_submaps_2d.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Matching", total=len(ground_submaps_2d))
        for ground_key, ground_sm_i in iterator:
            results_matrix = PoseEstimationResultMatrix(
                (aerial_x_max + 1, aerial_y_max + 1)
            )
            details_for_ground = {}

            ground_pose_ref = None
            if reference_trajectory is not None:
                ground_pose_ref = reference_trajectory.pose(
                    ground_submaps[ground_key].time
                )
            T_ground_odom_ground_robot = ground_sm_i.metadata["camera_pose"]

            for aerial_key, aerial_sm_j in aerial_submaps_2d.items():
                i_a, j_a = aerial_key_to_tuple(aerial_key)

                # --- Mode-based filtering ---
                filtered_keys = self._get_filtered_aerial_keys(
                    mode,
                    ground_key,
                    aerial_key,
                    i_a,
                    j_a,
                    aerial_sm_j,
                    ground_pose_ref,
                    stride_m,
                    patch_size_m_px,
                    vpr_top_k_sets,
                    aerial_submaps_2d,
                    ground_sm_i,
                    reference_trajectory,
                )
                if not filtered_keys:
                    continue

                single_result = self._match_single_pair(
                    aerial_sm_j,
                    ground_sm_i,
                    reference_trajectory,
                    T_camera_flu,
                    ground_pose_ref,
                    T_ground_odom_ground_robot,
                    do_translation_only,
                    local_to_pixel_fn=local_to_pixel_fn,
                )

                results_matrix[i_a, j_a] = single_result.pose_result
                details_for_ground[aerial_key] = single_result

            all_results[ground_key] = results_matrix
            all_details[ground_key] = details_for_ground

        return CrossViewMatchResult(results=all_results, match_details=all_details)

    # ------------------------------------------------------------------
    # Helpers for cross_view_match
    # ------------------------------------------------------------------

    def _preprocess_submaps_2d(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
    ) -> Tuple[Dict[str, Submap], Dict[str, Submap]]:
        """Convert submaps to 2D and filter short lines."""
        aerial_submaps_2d: Dict[str, Submap] = {}
        ground_submaps_2d: Dict[str, Submap] = {}
        for submaps_2d_dict, original_submaps_dict in [
            (aerial_submaps_2d, aerial_submaps),
            (ground_submaps_2d, ground_submaps),
        ]:
            for key, submap in original_submaps_dict.items():
                segments_2d = submap.segments.to_dim(2)
                filtered_lines = [
                    line
                    for line in segments_2d.get_lines()
                    if line.get_length() >= self.pipeline_params.match_min_len_m
                ]
                segments_2d = segments_2d.get_points() + SegmentList(filtered_lines)
                submaps_2d_dict[key] = deepcopy(submap)
                submaps_2d_dict[key].segments = segments_2d

        # Transfer height for ground submaps
        for ground_key in ground_submaps_2d.keys():
            for segment in ground_submaps_2d[ground_key].segments:
                segment.height = (
                    ground_submaps[ground_key]
                    .segments.get_segment_from_id(segment.id)
                    .height
                )

        return aerial_submaps_2d, ground_submaps_2d

    def _get_filtered_aerial_keys(
        self,
        mode: str,
        ground_key: str,
        aerial_key: str,
        i_a: int,
        j_a: int,
        aerial_sm_j: Submap,
        ground_pose_ref,
        stride_m: float,
        patch_size_m_px: float,
        vpr_top_k_sets: dict,
        aerial_submaps_2d: Dict[str, Submap],
        ground_sm_i: Submap,
        reference_trajectory,
    ) -> bool:
        """Return True if this aerial key should be processed for the given mode."""
        if mode == "gt":
            if ground_pose_ref is None:
                # Fall back to VPR
                if (i_a, j_a) not in vpr_top_k_sets.get(ground_key, set()):
                    return False
            else:
                T_aerial_camera = np.linalg.inv(aerial_sm_j.pose) @ ground_pose_ref
                ground_pos_aerial = T_aerial_camera[:2, 3]
                x1_m = i_a * stride_m
                y1_m = j_a * stride_m
                x2_m = x1_m + patch_size_m_px
                y2_m = y1_m + patch_size_m_px
                if not (
                    x1_m <= ground_pos_aerial[0] <= x2_m
                    and y1_m <= ground_pos_aerial[1] <= y2_m
                ):
                    return False
        elif mode == "vpr":
            if (i_a, j_a) not in vpr_top_k_sets.get(ground_key, set()):
                return False
        elif mode == "max_intersection":
            # Handled at a higher level — we process all patches and pick the best.
            # The caller should use _find_max_intersection_patch instead.
            pass
        # mode == "all": no filtering
        return True

    def _match_single_pair(
        self,
        aerial_sm: Submap,
        ground_sm: Submap,
        reference_trajectory,
        T_camera_flu: np.ndarray,
        ground_pose_ref,
        T_ground_odom_ground_robot: np.ndarray,
        translation_only: bool,
        local_to_pixel_fn=None,
    ) -> SingleMatchResult:
        """Match a single aerial-ground submap pair."""
        # Compute rotation from reference trajectory if available
        if ground_pose_ref is not None:
            T_aerial_camera = np.linalg.inv(aerial_sm.pose) @ ground_pose_ref
            T_aerial_odom = T_aerial_camera @ np.linalg.inv(T_ground_odom_ground_robot)
            R_aerial_ground_2d = T_aerial_odom[:2, :2]
            if T_camera_flu is not None:
                T_aerial_ground = T_aerial_camera @ T_camera_flu
            else:
                T_aerial_ground = T_aerial_camera
            # T_aerial_ground[:2, 3] is in UTM-delta frame (inv(pose_flu_UTM) @ UTM_pose).
            # T_aerial_ground_hat will be in pixel-meter frame (from segment registerer).
            # Convert GT translation to pixel-meter frame so errors are computed correctly.
            if local_to_pixel_fn is not None:
                pixel_len_m = self.aerial_segmenter.params.pixel_len_m
                col, row = local_to_pixel_fn(
                    float(T_aerial_ground[0, 3]), float(T_aerial_ground[1, 3])
                )
                T_aerial_ground = T_aerial_ground.copy()
                T_aerial_ground[0, 3] = col * pixel_len_m
                T_aerial_ground[1, 3] = row * pixel_len_m
        else:
            T_aerial_ground = np.zeros((4, 4)) * np.nan
            R_aerial_ground_2d = None

        # Split long lines before matching
        ground_segs_i = ground_sm.segments.get_points() + split_long_lines(
            ground_sm.segments.get_lines(),
            max_length=self.pipeline_params.line_split_length_m,
        )
        ground_segs_i.reindex()
        aerial_segs_j = aerial_sm.segments.get_points() + split_long_lines(
            aerial_sm.segments.get_lines(),
            max_length=self.pipeline_params.line_split_length_m,
        )
        aerial_segs_j.reindex()

        if self.pipeline_params.points_only:
            ground_segs_i = ground_segs_i.get_points()
            aerial_segs_j = aerial_segs_j.get_points()
        elif self.pipeline_params.lines_only:
            ground_segs_i = ground_segs_i.get_lines()
            aerial_segs_j = aerial_segs_j.get_lines()

        match_kwargs = {}
        if translation_only and R_aerial_ground_2d is not None:
            # Apply rotation perturbation if configured
            noise_deg = self.pipeline_params.rot_bias_deg
            lo, hi = self.pipeline_params.uniform_rot_noise_bounds_deg
            if lo != 0.0 or hi != 0.0:
                noise_deg += np.random.uniform(lo, hi)
            if noise_deg != 0.0:
                noise_rad = np.deg2rad(noise_deg)
                R_noise = np.array(
                    [
                        [np.cos(noise_rad), -np.sin(noise_rad)],
                        [np.sin(noise_rad), np.cos(noise_rad)],
                    ]
                )
                R_aerial_ground_2d = R_noise @ R_aerial_ground_2d

            # Aerial is axis-aligned
            match_kwargs["global_x_dir1"] = np.array([1.0, 0.0])
            match_kwargs["global_y_dir1"] = np.array([0.0, 1.0])
            # Ground dirs from (possibly perturbed) rotation
            match_kwargs["global_x_dir2"] = R_aerial_ground_2d[:, 0]
            match_kwargs["global_y_dir2"] = R_aerial_ground_2d[:, 1]

            # Ensure the matcher uses the direction constraints
            self.matcher.params.xy_dir_constrained_2d = True

        matches = self.matcher.match(
            aerial_segs_j,
            ground_segs_i,
            **match_kwargs,
        )

        # Restore default so non-translation-only calls are unaffected
        if translation_only and R_aerial_ground_2d is not None:
            self.matcher.params.xy_dir_constrained_2d = False
        try:
            T_aerial_ground_odom_hat = self.registerer.register(
                aerial_segs_j.to_dim(3),
                ground_segs_i.to_dim(3),
                correspondences=matches,
            ).transformation
            T_aerial_ground_hat = T_aerial_ground_odom_hat @ T_ground_odom_ground_robot
            if T_camera_flu is not None:
                T_aerial_ground_hat = T_aerial_ground_hat @ T_camera_flu
        except InsufficientAssociationsException:
            T_aerial_ground_hat = np.zeros((4, 4)) * np.nan

        # no z component estimated
        T_aerial_ground[2, 3] = 0.0
        if not np.any(np.isnan(T_aerial_ground_hat)):
            T_aerial_ground_hat[2, 3] = 0.0

        pose_result = PoseEstimationResult(
            T_i_j_hat=T_aerial_ground_hat,
            T_i_j=T_aerial_ground,
            associations=matches,
        )

        matched_ground = SegmentList(
            [ground_segs_i.get_segment_from_id(g_id) for g_id, _ in matches]
        )
        matched_aerial = SegmentList(
            [aerial_segs_j.get_segment_from_id(a_id) for _, a_id in matches]
        )

        return SingleMatchResult(
            pose_result=pose_result,
            aerial_segs_processed=aerial_segs_j,
            ground_segs_processed=ground_segs_i,
            matched_ground=matched_ground,
            matched_aerial=matched_aerial,
        )

    def _find_max_intersection_patch(
        self,
        ground_submap: Submap,
        aerial_submaps_2d: Dict[str, Submap],
        reference_trajectory,
        T_camera_flu: np.ndarray = None,
    ) -> Optional[str]:
        """Find the aerial patch with maximum overlap with the ground submap's
        estimated position circle.

        Uses shapely to compute circle-rectangle intersection area.
        Returns the aerial key of the best patch, or None if no overlap.
        """
        if reference_trajectory is None:
            return None

        try:
            from shapely.geometry import Point, box
        except ImportError:
            logger.warning("shapely not installed; max_intersection mode unavailable.")
            return None

        ground_pose = reference_trajectory.pose(ground_submap.time)

        # Ground position in aerial frame (pick any aerial submap for pose)
        first_key = next(iter(aerial_submaps_2d))
        aerial_pose = aerial_submaps_2d[first_key].pose
        if T_camera_flu is not None:
            T_aerial_ground = np.linalg.inv(aerial_pose) @ ground_pose @ T_camera_flu
        else:
            T_aerial_ground = np.linalg.inv(aerial_pose) @ ground_pose
        gx, gy = T_aerial_ground[0, 3], T_aerial_ground[1, 3]

        rad_m = self.pipeline_params.ground_submap_rad_m
        ground_circle = Point(gx, gy).buffer(rad_m)

        # Compute crop geometry
        pixel_len_m = self.aerial_segmenter.params.pixel_len_m
        px_per_m = 1.0 / pixel_len_m
        patch_size_px = int(self.pipeline_params.aerial_img_patch_side_len_m * px_per_m)
        stride = int(
            patch_size_px * (1.0 - self.pipeline_params.aerial_img_patch_overlap)
        )
        stride_m = stride * pixel_len_m
        patch_size_m = patch_size_px * pixel_len_m

        best_key = None
        best_area = 0.0
        aerial_key_to_tuple = _aerial_key_to_tuple

        for aerial_key in aerial_submaps_2d:
            i_a, j_a = aerial_key_to_tuple(aerial_key)
            x1_m = i_a * stride_m
            y1_m = j_a * stride_m
            x2_m = x1_m + patch_size_m
            y2_m = y1_m + patch_size_m
            patch_rect = box(x1_m, y1_m, x2_m, y2_m)
            area = ground_circle.intersection(patch_rect).area
            if area > best_area:
                best_area = area
                best_key = aerial_key

        return best_key

    def cross_view_match_max_intersection(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        reference_trajectory=None,
        T_camera_flu: np.ndarray = None,
        translation_only: bool = None,
        show_progress: bool = False,
        local_to_pixel_fn=None,
    ) -> CrossViewMatchResult:
        """Cross-view match using max_intersection mode.

        For each ground submap, finds the single best-overlapping aerial patch
        and matches only against that one.
        """
        do_translation_only = (
            translation_only
            if translation_only is not None
            else self.pipeline_params.translation_only
        )

        aerial_submaps_2d, ground_submaps_2d = self._preprocess_submaps_2d(
            aerial_submaps, ground_submaps
        )

        aerial_key_to_tuple = _aerial_key_to_tuple
        aerial_x_max = max(aerial_key_to_tuple(key)[0] for key in aerial_submaps.keys())
        aerial_y_max = max(aerial_key_to_tuple(key)[1] for key in aerial_submaps.keys())

        all_results = {}
        all_details = {}

        iterator = ground_submaps_2d.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Matching", total=len(ground_submaps_2d))
        for ground_key, ground_sm_i in iterator:
            results_matrix = PoseEstimationResultMatrix(
                (aerial_x_max + 1, aerial_y_max + 1)
            )
            details_for_ground = {}

            best_aerial_key = self._find_max_intersection_patch(
                ground_submaps[ground_key],
                aerial_submaps_2d,
                reference_trajectory,
                T_camera_flu,
            )

            if best_aerial_key is None:
                all_results[ground_key] = results_matrix
                all_details[ground_key] = details_for_ground
                continue

            aerial_sm_j = aerial_submaps_2d[best_aerial_key]
            i_a, j_a = aerial_key_to_tuple(best_aerial_key)

            ground_pose_ref = None
            if reference_trajectory is not None:
                ground_pose_ref = reference_trajectory.pose(
                    ground_submaps[ground_key].time
                )
            T_ground_odom_ground_robot = ground_sm_i.metadata["camera_pose"]

            single_result = self._match_single_pair(
                aerial_sm_j,
                ground_sm_i,
                reference_trajectory,
                T_camera_flu,
                ground_pose_ref,
                T_ground_odom_ground_robot,
                do_translation_only,
                local_to_pixel_fn=local_to_pixel_fn,
            )

            results_matrix[i_a, j_a] = single_result.pose_result
            details_for_ground[best_aerial_key] = single_result

            all_results[ground_key] = results_matrix
            all_details[ground_key] = details_for_ground

        return CrossViewMatchResult(results=all_results, match_details=all_details)
