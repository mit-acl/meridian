import io
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
import cv2 as cv
import os
import argparse
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import pickle

from meridian.params import (
    PrimitiveMatchParams,
    CrossViewMatchingParams,
    CrossViewVisualizationParams,
    CrossViewLocalizationDataParams,
    CrossViewPlaceRecognitionParams,
    AerialSegmenterParams,
    RegisterParams,
    SegmentToPrimitiveConversionParams,
    AerialPatchParams,
    GroundSubmapParams,
    GroundSegmenterParams,
)
from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
from meridian.pipeline.data import CrossViewLocalizationData
from meridian.utils import expandvars_recursive
from meridian.segmenter.aerial_segmenter import AerialSegmenter
from meridian.map2d.segment_to_primitive import SegmentToPrimitiveConverter
from meridian.map2d.aerial_patch_primitive_mapping import (
    AerialPatchPrimitiveMapping,
    AerialSegmentationResult,
)
from meridian.map2d.ground_submap_primitive_mapping import (
    GroundSubmapPrimitiveMapping,
    GroundSegmentationResult,
)
from meridian.match.primitive_matcher import PrimitiveMatcher
from meridian.register.registerer import Registerer2D
from meridian.map3d.submap import Submap
from meridian.viz.cross_view_viz import (
    viz_cross_view_matches,
    viz_registration_alignment,
    viz_alignment_fitness,
    viz_aerial_segments,
    viz_general_segments_img,
    viz_ground_segments,
    downsample_to_target_size,
)

# Re-export algorithm class for backward compatibility
from meridian.cross_view.matching import (  # noqa: F401
    CrossViewMatching,
    CrossViewMatchResult,
    SingleMatchResult,
)

Crop = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Module-level worker functions for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------


def _aerial_viz_worker(
    patch_img,
    aerial_segments,
    general_segments,
    sparse_segments,
    crop,
    pixel_len_m,
    alpha_shape_alpha,
    alpha_shape_grid_downsample,
    alpha_shape_max_n_pts,
    alpha_shape_ref_size_m,
    aerial_viz_downsample,
    aerial_viz_line_width_m,
    aerial_viz_target_size_kb,
    px_per_m,
    i,
    j,
):
    """Render 3 aerial viz images for one patch. Returns list of (filename, bytes)."""
    results = []

    aerial_viz = viz_aerial_segments(
        patch_img,
        aerial_segments,
        crop,
        pixel_len_m,
        alpha_shape_alpha,
        alpha_shape_grid_downsample,
        alpha_shape_max_n_pts,
        alpha_shape_ref_size_m,
        aerial_viz_downsample,
        line_width_m=aerial_viz_line_width_m,
    )
    viz_bytes = downsample_to_target_size(aerial_viz, aerial_viz_target_size_kb)
    results.append((f"{i}_{j}_segments.jpg", viz_bytes))

    general_viz = viz_general_segments_img(
        patch_img,
        general_segments,
        crop=crop,
        px_per_m=px_per_m,
        downsample_factor=aerial_viz_downsample,
        line_width_m=aerial_viz_line_width_m,
    )
    viz_bytes = downsample_to_target_size(general_viz, aerial_viz_target_size_kb)
    results.append((f"{i}_{j}_fine.jpg", viz_bytes))

    sparse_viz = viz_general_segments_img(
        patch_img,
        sparse_segments,
        crop=crop,
        px_per_m=px_per_m,
        downsample_factor=aerial_viz_downsample,
        line_width_m=aerial_viz_line_width_m,
    )
    viz_bytes = downsample_to_target_size(sparse_viz, aerial_viz_target_size_kb)
    results.append((f"{i}_{j}_sparse.jpg", viz_bytes))

    return results


def _ground_viz_worker(
    flattened_submap,
    aerial_segments,
    general_segments,
    sparse_segments,
    alpha_shape_alpha,
    alpha_shape_grid_downsample,
    alpha_shape_max_n_pts,
    alpha_shape_ref_size_m,
    show_sm_origin,
):
    """Render ground viz for one submap. Returns PNG bytes."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = viz_ground_segments(
        flattened_submap,
        aerial_segments,
        general_segments,
        sparse_segments,
        alpha_shape_alpha,
        alpha_shape_grid_downsample,
        alpha_shape_max_n_pts,
        alpha_shape_ref_size_m,
        show_origin=show_sm_origin,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=400)
    plt.close(fig)
    return buf.getvalue()


def _match_viz_worker(
    aerial_segs_processed,
    ground_segs_processed,
    associations,
    aerial_crop,
    ground_segments_all,
    dense_points_by_id,
    px_per_m,
    aerial_origin_m,
    match_viz_target_size_kb,
    aerial_img_crop,
    crop_origin_px,
    T_i_j,
    T_i_j_hat,
    patch_size_px,
    aerial_viz_target_size_kb,
    line_width_px,
    T_align=None,
    fitness_result=None,
    T_aerial_ground_2d=None,
):
    """Render match + pose viz for one ground-aerial pair. Returns list of (tag, bytes)."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = []

    # Match viz
    viz_cross_view_matches(
        aerial_segs_processed,
        ground_segs_processed,
        associations,
        aerial_crop=aerial_crop,
        ground_segments_all=ground_segments_all,
        dense_points_by_id=dense_points_by_id,
        px_per_m=px_per_m,
        aerial_origin_m=aerial_origin_m,
    )
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_array = cv.imdecode(
        np.frombuffer(buf.getvalue(), dtype=np.uint8), cv.IMREAD_COLOR
    )
    viz_bytes = downsample_to_target_size(img_array, match_viz_target_size_kb)
    results.append(("match", viz_bytes))

    # Pose viz — draw directly on pre-cropped image
    if aerial_img_crop is not None:
        crop = aerial_img_crop.copy()
        if len(crop.shape) == 2:
            crop = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
        x1, y1 = crop_origin_px
        arrow_len = 0.08 * patch_size_px

        def _draw_pose(img, T, color):
            pos_px = T[:2, 3] * px_per_m - np.array([x1, y1], dtype=float)
            cx, cy = int(round(pos_px[0])), int(round(pos_px[1]))
            cv.circle(img, (cx, cy), 5, color, -1)
            x_dir = T[:2, 0]
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)
            y_dir = T[:2, 1]
            y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)
            x_end = (
                int(round(cx + arrow_len * x_dir[0])),
                int(round(cy + arrow_len * x_dir[1])),
            )
            y_end = (
                int(round(cx + arrow_len * y_dir[0])),
                int(round(cy + arrow_len * y_dir[1])),
            )
            cv.arrowedLine(img, (cx, cy), x_end, color, line_width_px, tipLength=0.3)
            cv.arrowedLine(img, (cx, cy), y_end, color, line_width_px, tipLength=0.3)

        _draw_pose(crop, T_i_j, (0, 200, 0))
        if T_i_j_hat is not None:
            _draw_pose(crop, T_i_j_hat, (0, 0, 220))
        viz_bytes = downsample_to_target_size(crop, aerial_viz_target_size_kb)
        results.append(("pose", viz_bytes))

    # Registration alignment viz: inlier ground (transformed) overlaid on aerial
    if (
        T_align is not None
        and not np.any(np.isnan(T_align))
        and associations is not None
        and len(associations) > 0
    ):
        viz_registration_alignment(
            aerial_segs_processed,
            ground_segs_processed,
            associations,
            T_align,
        )
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_array = cv.imdecode(
            np.frombuffer(buf.getvalue(), dtype=np.uint8), cv.IMREAD_COLOR
        )
        viz_bytes = downsample_to_target_size(img_array, match_viz_target_size_kb)
        results.append(("reg", viz_bytes))

    if (
        fitness_result is not None
        and T_aerial_ground_2d is not None
        and not np.any(np.isnan(T_aerial_ground_2d))
    ):
        viz_alignment_fitness(
            aerial_segs_processed,
            ground_segs_processed,
            fitness_result,
            T_aerial_ground_2d,
            aerial_crop_img=aerial_crop,
        )
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_array = cv.imdecode(
            np.frombuffer(buf.getvalue(), dtype=np.uint8), cv.IMREAD_COLOR
        )
        viz_bytes = downsample_to_target_size(img_array, match_viz_target_size_kb)
        results.append(("fitness", viz_bytes))

    return results


# ---------------------------------------------------------------------------
# CrossViewMatchingPipeline — thin I/O wrapper
# ---------------------------------------------------------------------------


@dataclass
class CrossViewMatchingPipeline:
    algorithm: CrossViewMatching
    aerial_mapping: AerialPatchPrimitiveMapping = None
    ground_mapping: GroundSubmapPrimitiveMapping = None
    _viz_params: CrossViewVisualizationParams = field(
        default_factory=CrossViewVisualizationParams
    )
    _conversion_params: SegmentToPrimitiveConversionParams = field(
        default_factory=SegmentToPrimitiveConversionParams
    )

    # ------------------------------------------------------------------
    # Delegated convenience accessors
    # ------------------------------------------------------------------

    def load_submaps_from_dir(
        self, submap_dir: Union[str, pathlib.Path]
    ) -> Dict[str, Submap]:
        submap_dir = pathlib.Path(submap_dir)
        segment_files = list(submap_dir.glob("*.pkl"))
        segment_files.sort()
        results = {}
        for segment_file in segment_files:
            submap = Submap.load(segment_file)
            results[segment_file.stem] = submap
        return results

    # ------------------------------------------------------------------
    # Aerial segmentation with I/O
    # ------------------------------------------------------------------

    def run_aerial(
        self,
        img: np.ndarray,
        output_dir: Union[str, pathlib.Path],
        img_origin: np.ndarray = None,
    ) -> Dict[Crop, Submap]:
        result = self.aerial_mapping.run(
            img, img_origin, return_intermediates=True, show_progress=True
        )
        self._save_aerial_results(result, output_dir)
        return result.submaps

    def _save_aerial_results(self, result: AerialSegmentationResult, output_dir):
        output_dir = pathlib.Path(output_dir)
        viz_output_dir = output_dir / "viz"
        segment_output_dir = output_dir / "segments"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        segment_output_dir.mkdir(parents=True, exist_ok=True)

        conv_params = self._conversion_params
        patch_params = self.aerial_mapping.patch_params
        pixel_len_m = self.aerial_mapping.aerial_segmenter.params.pixel_len_m
        px_per_m = 1.0 / pixel_len_m

        max_workers = conv_params.sparse_conversion_max_threads
        patch_size_px = int(patch_params.aerial_img_patch_side_len_m * px_per_m)
        stride = int(patch_size_px * (1.0 - patch_params.aerial_img_patch_overlap))

        # Save submaps (fast I/O, keep serial)
        crop_ij = {}
        for crop, submap in result.submaps.items():
            x1, y1, x2, y2 = crop
            i = x1 // stride
            j = y1 // stride
            crop_ij[crop] = (i, j)
            fname_segment = segment_output_dir / f"{i}_{j}.pkl"
            submap.save(fname_segment)

        # Parallel visualization
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for crop, submap in result.submaps.items():
                if result.intermediates is None or crop not in result.intermediates:
                    continue
                intermediates = result.intermediates[crop]
                i, j = crop_ij[crop]
                future = executor.submit(
                    _aerial_viz_worker,
                    intermediates.patch_img,
                    intermediates.aerial_segments,
                    intermediates.general_segments,
                    submap.segments,
                    crop,
                    pixel_len_m,
                    conv_params.alpha_shape_alpha,
                    conv_params.alpha_shape_grid_downsample,
                    conv_params.alpha_shape_max_n_pts,
                    conv_params.alpha_shape_ref_size_m,
                    max(
                        1, round(self._viz_params.aerial_viz_pixel_size_m / pixel_len_m)
                    ),
                    self._viz_params.aerial_viz_line_width_m,
                    self._viz_params.aerial_viz_target_size_kb,
                    px_per_m,
                    i,
                    j,
                )
                futures[future] = crop

            for future in as_completed(futures):
                for fname, viz_bytes in future.result():
                    with open(str(viz_output_dir / fname), "wb") as f:
                        f.write(viz_bytes)

    # ------------------------------------------------------------------
    # Ground segmentation with I/O
    # ------------------------------------------------------------------

    def run_ground(
        self,
        submaps: List[Submap],
        output_dir: Union[str, pathlib.Path],
    ) -> List[Submap]:
        result = self.ground_mapping.batch_convert(
            submaps, return_intermediates=True, show_progress=True
        )
        self._save_ground_results(result, output_dir)
        return result.submaps

    def _save_ground_results(self, result: GroundSegmentationResult, output_dir):
        output_dir = pathlib.Path(output_dir)
        viz_output_dir = output_dir / "viz"
        segment_output_dir = output_dir / "segments"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        segment_output_dir.mkdir(parents=True, exist_ok=True)

        conv_params = self._conversion_params
        ground_params = self.ground_mapping.submap_params

        max_workers = conv_params.sparse_conversion_max_threads
        futures = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for k, submap_2d in enumerate(result.submaps):
                # Save submap (fast I/O, keep serial)
                fname_segment = segment_output_dir / f"{k}.pkl"
                submap_2d.save(fname_segment)

                # Save per-segment dense 2D points (fast I/O, keep serial)
                if result.intermediates is not None:
                    intermediates = result.intermediates[k]
                    dense_dir = segment_output_dir / f"{k}_dense"
                    dense_dir.mkdir(parents=True, exist_ok=True)
                    for aerial_seg in intermediates.aerial_segments:
                        dense_path = dense_dir / f"{aerial_seg.id}.pkl"
                        pts = aerial_seg.points
                        max_n = ground_params.dense_points_max_n
                        if max_n is not None and len(pts) > max_n:
                            idx = np.round(np.linspace(0, len(pts) - 1, max_n)).astype(
                                int
                            )
                            pts = pts[idx]
                        with open(dense_path, "wb") as f:
                            pickle.dump(pts, f)

                    # Submit visualization to pool
                    future = executor.submit(
                        _ground_viz_worker,
                        intermediates.flattened_submap,
                        intermediates.aerial_segments,
                        intermediates.general_segments,
                        submap_2d.segments,
                        conv_params.alpha_shape_alpha,
                        conv_params.alpha_shape_grid_downsample,
                        conv_params.alpha_shape_max_n_pts,
                        conv_params.alpha_shape_ref_size_m,
                        ground_params.viz_show_sm_origin,
                    )
                    futures[future] = k

            for future in as_completed(futures):
                k = futures[future]
                png_bytes = future.result()
                fname_general = viz_output_dir / f"{k}.png"
                with open(str(fname_general), "wb") as f:
                    f.write(png_bytes)

    # ------------------------------------------------------------------
    # Cross-view matching with I/O
    # ------------------------------------------------------------------

    def run_match(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        reference_trajectory=None,
        output_dir: Union[str, pathlib.Path] = None,
        aerial_img: np.ndarray = None,
        T_camera_flu: np.ndarray = None,
        matching_mode: str = None,
        translation_only: bool = None,
        ground_dense_dir: pathlib.Path = None,
        local_to_pixel_fn=None,
        save_viz: bool = True,
        gt_trajectory=None,
    ) -> CrossViewMatchResult:
        mode = matching_mode or self.algorithm.pipeline_params.matching_mode
        if mode == "max_intersection":
            match_result = self.algorithm.cross_view_match_max_intersection(
                aerial_submaps,
                ground_submaps,
                reference_trajectory,
                T_camera_flu,
                translation_only,
                show_progress=True,
                local_to_pixel_fn=local_to_pixel_fn,
                gt_trajectory=gt_trajectory,
            )
        else:
            match_result = self.algorithm.cross_view_match(
                aerial_submaps,
                ground_submaps,
                reference_trajectory,
                T_camera_flu,
                matching_mode,
                translation_only,
                show_progress=True,
                local_to_pixel_fn=local_to_pixel_fn,
                gt_trajectory=gt_trajectory,
            )
        if output_dir is not None:
            self._save_match_results(
                match_result,
                output_dir,
                aerial_img,
                aerial_submaps,
                ground_submaps,
                ground_dense_dir,
                local_to_pixel_fn=local_to_pixel_fn,
                save_viz=save_viz,
                T_camera_flu=T_camera_flu,
            )
        return match_result

    def _save_match_results(
        self,
        match_result: CrossViewMatchResult,
        output_dir,
        aerial_img: np.ndarray,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        ground_dense_dir: pathlib.Path = None,
        local_to_pixel_fn=None,
        save_viz: bool = True,
        T_camera_flu: np.ndarray = None,
    ):
        output_dir = pathlib.Path(output_dir)
        viz_output_dir = output_dir / "viz"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        segments_output_dir = output_dir / "segments"
        segments_output_dir.mkdir(parents=True, exist_ok=True)

        params = self.algorithm.pipeline_params
        patch_params = self.algorithm.aerial_patch_params
        pixel_len_m = self.algorithm.pixel_len_m
        px_per_m = 1.0 / pixel_len_m
        patch_size_px = int(patch_params.aerial_img_patch_side_len_m * px_per_m)
        stride = int(patch_size_px * (1.0 - patch_params.aerial_img_patch_overlap))

        def aerial_key_to_tuple(key):
            return tuple(int(x) for x in key.split("_"))

        # Phase 1: serial I/O — save segments, load dense points, save heatmaps
        dense_points_all = {}  # ground_key -> dense_points_by_id
        for ground_key, results_matrix in match_result.results.items():
            ground_sub_dir = viz_output_dir / f"ground_{ground_key}"
            ground_sub_dir.mkdir(parents=True, exist_ok=True)

            # Load per-segment dense 2D points for this ground submap
            if ground_dense_dir is not None:
                dense_dir = pathlib.Path(ground_dense_dir) / f"{ground_key}_dense"
            else:
                dense_dir = (
                    output_dir.parent / "ground" / "segments" / f"{ground_key}_dense"
                )
            dense_points_by_id = {}
            if dense_dir.exists():
                for dense_file in dense_dir.glob("*.pkl"):
                    seg_id = int(dense_file.stem)
                    with open(dense_file, "rb") as f:
                        dense_points_by_id[seg_id] = pickle.load(f)
            dense_points_all[ground_key] = dense_points_by_id

            details = match_result.match_details.get(ground_key, {})

            # Save matched segments
            for aerial_key, single_results_list in details.items():
                # match_details now holds List[SingleMatchResult] per pair;
                # save only the primary (best) hypothesis.
                if isinstance(single_results_list, list):
                    single_result = single_results_list[0]
                else:
                    single_result = single_results_list
                fname_matches = (
                    segments_output_dir / f"ground_{ground_key}_aerial_{aerial_key}.pkl"
                )
                with open(fname_matches, "wb") as f:
                    pickle.dump(
                        [single_result.matched_ground, single_result.matched_aerial],
                        f,
                    )

            # Save results matrix and heatmap
            results_matrix.save(
                segments_output_dir / f"ground_{ground_key}_results_matrix.pkl"
            )

            # Compute GT patch indices from T_i_j
            gt_patches = None
            for idx in np.ndindex(results_matrix.shape):
                cell = results_matrix[idx]
                primary = cell[0] if isinstance(cell, list) else cell
                t = primary.T_i_j
                if not np.any(np.isnan(t)):
                    ground_pos = t[:2, 3]
                    gt_patches = []
                    stride_m = stride * pixel_len_m
                    patch_size_m = patch_params.aerial_img_patch_side_len_m
                    for i_a in range(results_matrix.shape[0]):
                        for j_a in range(results_matrix.shape[1]):
                            x1 = i_a * stride_m
                            y1 = j_a * stride_m
                            if (
                                x1 <= ground_pos[0] <= x1 + patch_size_m
                                and y1 <= ground_pos[1] <= y1 + patch_size_m
                            ):
                                gt_patches.append((i_a, j_a))
                    break

            results_matrix.plot_cross_view(
                dist_thresh=params.match_trans_err_m,
                angle_thresh_deg=params.match_rot_err_deg,
                gt_patches=gt_patches,
            )
            fname_heatmap = viz_output_dir / f"ground_{ground_key}_all.png"
            plt.savefig(fname_heatmap, dpi=400)
            plt.close()

        # Phase 2: parallel match + pose visualization across ALL ground keys
        if save_viz:
            max_workers = self._conversion_params.sparse_conversion_max_threads
            line_width_px = max(
                1, round(self._viz_params.aerial_viz_line_width_m * px_per_m)
            )
            viz_futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for ground_key in match_result.results:
                    details = match_result.match_details.get(ground_key, {})
                    dense_points_by_id = dense_points_all[ground_key]

                    for aerial_key, single_results_list in details.items():
                        if isinstance(single_results_list, list):
                            single_result = single_results_list[0]
                        else:
                            single_result = single_results_list
                        i_a, j_a = aerial_key_to_tuple(aerial_key)
                        result = single_result.pose_result

                        aerial_crop = None
                        aerial_origin_m = None
                        if aerial_img is not None:
                            x1_a = i_a * stride
                            y1_a = j_a * stride
                            aerial_crop = aerial_img[
                                y1_a : y1_a + patch_size_px,
                                x1_a : x1_a + patch_size_px,
                            ].copy()
                            aerial_origin_m = (x1_a / px_per_m, y1_a / px_per_m)

                        ground_segments_all = None
                        camera_pose = None
                        if ground_key in ground_submaps:
                            ground_segments_all = ground_submaps[ground_key].segments
                            camera_pose = ground_submaps[ground_key].metadata.get(
                                "camera_pose"
                            )

                        # Prepare pose viz args (pre-crop aerial image)
                        aerial_img_crop = None
                        crop_origin_px = None
                        T_i_j_val = result.T_i_j
                        T_i_j_hat_val = None
                        if aerial_img is not None and not np.any(
                            np.isnan(result.T_i_j)
                        ):
                            x1 = i_a * stride
                            y1 = j_a * stride
                            # Pre-crop aerial image for pose viz worker
                            aerial_img_crop = aerial_img[
                                y1 : y1 + patch_size_px,
                                x1 : x1 + patch_size_px,
                            ].copy()
                            crop_origin_px = (x1, y1)
                            T_i_j_hat_val = (
                                result.T_i_j_hat
                                if not np.any(np.isnan(result.T_i_j_hat))
                                else None
                            )

                        # Build T_align so ground_segs_processed (in odom frame) can be
                        # transformed into the aerial frame for the registration viz.
                        # T_i_j_hat = T_aerial_ground_odom_hat @ camera_pose [@ T_camera_flu]
                        T_align = None
                        if T_i_j_hat_val is not None and camera_pose is not None:
                            T_right = camera_pose
                            if T_camera_flu is not None:
                                T_right = T_right @ T_camera_flu
                            T_align = T_i_j_hat_val @ np.linalg.inv(T_right)

                        future = executor.submit(
                            _match_viz_worker,
                            single_result.aerial_segs_processed,
                            single_result.ground_segs_processed,
                            result.associations,
                            aerial_crop,
                            ground_segments_all,
                            dense_points_by_id,
                            px_per_m,
                            aerial_origin_m,
                            self._viz_params.match_viz_target_size_kb,
                            aerial_img_crop,
                            crop_origin_px,
                            T_i_j_val,
                            T_i_j_hat_val,
                            patch_size_px,
                            self._viz_params.aerial_viz_target_size_kb,
                            line_width_px,
                            T_align,
                            single_result.alignment_fitness,
                            single_result.T_aerial_ground_odom_2d,
                        )
                        viz_futures[future] = (ground_key, aerial_key)

                for future in as_completed(viz_futures):
                    gk, ak = viz_futures[future]
                    ground_sub_dir = viz_output_dir / f"ground_{gk}"
                    for tag, viz_bytes in future.result():
                        if tag == "match":
                            fname = ground_sub_dir / f"ground_{gk}_aerial_{ak}.jpg"
                        elif tag == "reg":
                            fname = ground_sub_dir / f"ground_{gk}_aerial_{ak}_reg.jpg"
                        elif tag == "fitness":
                            fname = ground_sub_dir / f"ground_{gk}_aerial_{ak}_fitness.jpg"
                        else:
                            fname = ground_sub_dir / f"ground_{gk}_aerial_{ak}_pose.jpg"
                        with open(fname, "wb") as f:
                            f.write(viz_bytes)

        params = self.algorithm.pipeline_params
        self._write_results_summary(
            match_result.results,
            output_dir,
            dist_thresh=params.match_trans_err_m,
            angle_thresh_deg=params.match_rot_err_deg,
        )

    @staticmethod
    def _write_results_summary(
        all_results, output_dir, dist_thresh=5.0, angle_thresh_deg=10.0
    ):
        n_total = len(all_results)
        n_best_success = 0
        n_max_assoc_success = 0
        n_any_hyp_success = 0
        best_ground_keys = []
        any_hyp_ground_keys = []

        for ground_key, results_matrix in all_results.items():
            trans_errors = results_matrix.translation_error_m
            rot_errors = np.rad2deg(results_matrix.rotation_error_rad)
            num_assoc = results_matrix.num_associations

            # Metric 1: best hypothesis in any crop below both thresholds
            success_mask = (trans_errors < dist_thresh) & (
                rot_errors < angle_thresh_deg
            )
            if np.any(success_mask):
                n_best_success += 1
                best_ground_keys.append(ground_key)

            # Metric 2: crop(s) with max associations — majority correct
            valid_mask = num_assoc > 0
            if np.any(valid_mask):
                max_assoc = np.nanmax(num_assoc[valid_mask])
                if max_assoc > 0:
                    tied_mask = num_assoc == max_assoc
                    tied_successes = np.sum(success_mask[tied_mask])
                    tied_total = np.sum(tied_mask)
                    if tied_successes > tied_total / 2:  # strict majority
                        n_max_assoc_success += 1

            # Metric 3: any hypothesis in any cell below both thresholds
            any_hyp_correct = False
            for idx in np.ndindex(results_matrix.shape):
                for h in results_matrix.all_hypotheses(idx):
                    t_err = h.translation_error_m
                    r_err = h.rotation_error_rad
                    if np.isnan(t_err) or np.isnan(r_err):
                        continue
                    if t_err < dist_thresh and np.rad2deg(r_err) < angle_thresh_deg:
                        any_hyp_correct = True
                        break
                if any_hyp_correct:
                    break
            if any_hyp_correct:
                n_any_hyp_success += 1
                any_hyp_ground_keys.append(ground_key)

        # Compute mean time per registration across all pairs
        all_runtimes = []
        for results_matrix in all_results.values():
            runtimes = results_matrix.runtime_s.flatten()
            all_runtimes.extend(runtimes[~np.isnan(runtimes)])
        mean_runtime = np.mean(all_runtimes) if all_runtimes else np.nan

        results_path = pathlib.Path(output_dir) / "results.txt"
        results_str = (
            f"Successful ground submap pose (best hypothesis): "
            f"{n_best_success} / {n_total}\n"
            f"Successful ground submap pose (max associations): "
            f"{n_max_assoc_success} / {n_total}\n"
            f"Successful ground submap pose (any hypothesis): "
            f"{n_any_hyp_success} / {n_total}\n"
            f"Mean time per registration: {mean_runtime:.3f} s\n"
            f"\nGround keys with successful registration "
            f"(best, {len(best_ground_keys)} / {n_total}): "
            + " ".join(best_ground_keys)
            + "\n"
            f"Ground keys with successful registration "
            f"(any, {len(any_hyp_ground_keys)} / {n_total}): "
            + " ".join(any_hyp_ground_keys)
            + "\n"
        )
        print(results_str)
        with open(results_path, "w") as f:
            f.write(results_str)


# ---------------------------------------------------------------------------
# Top-level entry point (backward-compatible signature)
# ---------------------------------------------------------------------------


def cross_view_matching(
    params,
    output_dir,
    skip_aerial=False,
    skip_ground=False,
    skip_match=False,
    save_viz=True,
    aerial_dir=None,
    ground_dir=None,
):
    pipeline_params = CrossViewMatchingParams.load(params)
    pipeline_params.output_directory = output_dir
    primitive_match_params = PrimitiveMatchParams.load(params)
    primitive_match_params.dim = 2

    conversion_params = SegmentToPrimitiveConversionParams.load(params)
    aerial_patch_params = AerialPatchParams.load(params)
    ground_submap_params = GroundSubmapParams.load(params)

    try:
        pr_params = CrossViewPlaceRecognitionParams.load(params)
    except Exception:
        pr_params = None
    if pr_params is None and pipeline_params.matching_mode == "vpr":
        pr_params = CrossViewPlaceRecognitionParams()
    place_recognition = CrossViewPlaceRecognition(pr_params) if pr_params else None

    aerial_segmenter = AerialSegmenter(AerialSegmenterParams.load(params))
    converter = SegmentToPrimitiveConverter(conversion_params)
    ground_segmenter_params = GroundSegmenterParams.load(params)

    aerial_mapping = AerialPatchPrimitiveMapping(
        patch_params=aerial_patch_params,
        converter=converter,
        aerial_segmenter=aerial_segmenter,
        place_recognition=place_recognition,
    )
    ground_mapping = GroundSubmapPrimitiveMapping(
        submap_params=ground_submap_params,
        converter=converter,
        ground_segmenter_params=ground_segmenter_params,
        place_recognition=place_recognition,
    )

    algorithm = CrossViewMatching(
        pipeline_params=pipeline_params,
        aerial_patch_params=aerial_patch_params,
        pixel_len_m=aerial_segmenter.params.pixel_len_m,
        matcher=PrimitiveMatcher(primitive_match_params),
        registerer=Registerer2D(RegisterParams.load(params)),
        place_recognition=place_recognition,
    )
    viz_params = CrossViewVisualizationParams.load(params)
    pipeline = CrossViewMatchingPipeline(
        algorithm=algorithm,
        aerial_mapping=aerial_mapping,
        ground_mapping=ground_mapping,
        _viz_params=viz_params,
        _conversion_params=conversion_params,
    )

    initial_aerial_segments = None
    initial_ground_segments = None

    # Load data
    data_params = CrossViewLocalizationDataParams.load(params)
    data = CrossViewLocalizationData.from_params(data_params)

    # Sync aerial segmenter pixel size with data
    aerial_segmenter.params.pixel_len_m = data.aerial_img_scale
    algorithm.pixel_len_m = data.aerial_img_scale

    # Set up output directories
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if aerial_dir is not None:
        aerial_output_dir = str(aerial_dir)
        skip_aerial = True
        with open(os.path.join(output_dir, "aerial.txt"), "w") as f:
            f.write(os.path.abspath(aerial_output_dir) + "\n")
    else:
        aerial_output_dir = os.path.join(output_dir, "aerial")

    if ground_dir is not None:
        ground_output_dir = str(ground_dir)
        skip_ground = True
        with open(os.path.join(output_dir, "ground.txt"), "w") as f:
            f.write(os.path.abspath(ground_output_dir) + "\n")
    else:
        ground_output_dir = os.path.join(output_dir, "ground")

    match_output_dir = os.path.join(output_dir, "match")

    # Save all params (including defaults) and commit hash
    from meridian.utils import save_params, save_commit_hash

    all_params = [
        data_params,
        pipeline_params,
        conversion_params,
        aerial_patch_params,
        ground_submap_params,
        primitive_match_params,
        algorithm.registerer.params,
        aerial_segmenter.params,
    ]
    if pr_params is not None:
        all_params.append(pr_params)
    save_params(output_dir, *all_params)
    save_commit_hash(output_dir)

    # Extract aerial segments
    if not skip_aerial:
        pipeline.run_aerial(data.aerial_img, aerial_output_dir, data.aerial_img_origin)

    # Extract ground segments
    if not skip_ground:
        if data.ground_map is None:
            raise ValueError(
                "Ground segmentation is enabled but no ground map is loaded. "
                "Set `ground_map_path` in the cross_view_localization_data params, "
                "or pass --ground <existing_ground_dir> to skip ground segmentation."
            )
        ground_map = data.ground_map
        ground_submaps = ground_mapping.create_submaps_from_map(ground_map)
        initial_ground_submaps = pipeline.run_ground(ground_submaps, ground_output_dir)
        initial_ground_segments = {
            str(k): segs for k, segs in enumerate(initial_ground_submaps)
        }

    if not skip_match:
        if initial_aerial_segments is None:
            initial_aerial_segments = pipeline.load_submaps_from_dir(
                os.path.join(aerial_output_dir, "segments")
            )
        if initial_ground_segments is None:
            initial_ground_segments = pipeline.load_submaps_from_dir(
                os.path.join(ground_output_dir, "segments")
            )
        pipeline.run_match(
            initial_aerial_segments,
            initial_ground_segments,
            data.gt_pose_data,
            match_output_dir,
            aerial_img=data.aerial_img,
            T_camera_flu=data.T_camera_flu,
            ground_dense_dir=os.path.join(ground_output_dir, "segments"),
            local_to_pixel_fn=data.aerial_local_to_pixel
            if data.geotiff_transform is not None
            else None,
            save_viz=save_viz,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to params directory or file. "
        + "Required params: cross_view_matching, cross_view_localization_data, "
        + "aerial_segmenter, segment_match, submap",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-aerial", action="store_true", help="Skip aerial segmentation."
    )
    parser.add_argument(
        "--skip-ground", action="store_true", help="Skip ground segmentation."
    )
    parser.add_argument(
        "--skip-match", action="store_true", help="Skip segment matching."
    )
    parser.add_argument(
        "-v",
        "--viz",
        action="store_true",
        help="Save per-match viz images (default: off).",
    )
    parser.add_argument(
        "--aerial",
        type=str,
        default=None,
        help="Path to existing aerial directory (skips aerial segmentation).",
    )
    parser.add_argument(
        "--ground",
        type=str,
        default=None,
        help="Path to existing ground directory (skips ground segmentation).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )

    if not os.path.isdir(args.output):
        os.mkdir(expandvars_recursive(args.output))

    cross_view_matching(
        args.params,
        args.output,
        args.skip_aerial,
        args.skip_ground,
        args.skip_match,
        save_viz=args.viz,
        aerial_dir=args.aerial,
        ground_dir=args.ground,
    )
