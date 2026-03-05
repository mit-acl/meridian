import io
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import cv2 as cv
import os
import argparse
import pathlib
import matplotlib.pyplot as plt
import pickle
import shutil

from gen_seg_match.params import (
    SegmentMatchParams,
    CrossViewMatchingParams,
    CrossViewLocalizationDataParams,
    CrossViewPlaceRecognitionParams,
    AerialSegmenterParams,
    SubmapParams,
    RegisterParams,
)
from gen_seg_match.cross_view.place_recognition import CrossViewPlaceRecognition
from gen_seg_match.pipeline.data import CrossViewLocalizationData
from gen_seg_match.utils import expandvars_recursive
from gen_seg_match.map2d.aerial_segmenter import AerialSegmenter
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.register.registerer import Registerer
from gen_seg_match.map3d.submap import Submap
from gen_seg_match.viz.cross_view_viz import (
    viz_cross_view_matches,
    viz_aerial_segments,
    viz_general_segments_img,
    viz_ground_segments,
    viz_pose_on_aerial_crop,
    downsample_to_target_size,
)

# Re-export algorithm class for backward compatibility
from gen_seg_match.cross_view.matching import (  # noqa: F401
    CrossViewMatching,
    CrossViewMatchResult,
    SingleMatchResult,
    AerialSegmentationResult,
    GroundSegmentationResult,
)

Crop = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# CrossViewMatchingPipeline — thin I/O wrapper
# ---------------------------------------------------------------------------


@dataclass
class CrossViewMatchingPipeline:
    algorithm: CrossViewMatching

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
        result = self.algorithm.batch_aerial_img_to_segments(
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

        params = self.algorithm.pipeline_params
        pixel_len_m = self.algorithm.aerial_segmenter.params.pixel_len_m
        px_per_m = 1.0 / pixel_len_m

        for crop, submap in result.submaps.items():
            x1, y1, x2, y2 = crop
            # Compute i, j from crop coordinates
            patch_size_px = int(params.aerial_img_patch_side_len_m * px_per_m)
            stride = int(patch_size_px * (1.0 - params.aerial_img_patch_overlap))
            i = x1 // stride
            j = y1 // stride

            # Save submap
            fname_segment = segment_output_dir / f"{i}_{j}.pkl"
            submap.save(fname_segment)

            # Save visualizations if intermediates are available
            if result.intermediates is not None and crop in result.intermediates:
                intermediates = result.intermediates[crop]

                # Raw AerialSegments overlay
                aerial_viz = viz_aerial_segments(
                    intermediates.patch_img,
                    intermediates.aerial_segments,
                    crop,
                    pixel_len_m,
                    params.alpha_shape_alpha,
                    params.alpha_shape_grid_downsample,
                    params.alpha_shape_max_n_pts,
                    params.alpha_shape_ref_size_m,
                    params.aerial_viz_downsample,
                    line_width_m=params.aerial_viz_line_width_m,
                )
                viz_bytes = downsample_to_target_size(
                    aerial_viz, params.aerial_viz_target_size_kb
                )
                fname_aerial = viz_output_dir / f"{i}_{j}_segments.jpg"
                with open(str(fname_aerial), "wb") as f:
                    f.write(viz_bytes)

                # GeneralSegments overlay (pre-sparsification)
                general_viz = viz_general_segments_img(
                    intermediates.patch_img,
                    intermediates.general_segments,
                    crop=crop,
                    px_per_m=px_per_m,
                    downsample_factor=params.aerial_viz_downsample,
                    line_width_m=params.aerial_viz_line_width_m,
                )
                viz_bytes = downsample_to_target_size(
                    general_viz, params.aerial_viz_target_size_kb
                )
                fname_general = viz_output_dir / f"{i}_{j}_fine.jpg"
                with open(str(fname_general), "wb") as f:
                    f.write(viz_bytes)

                # Sparse GeneralSegments overlay
                sparse_general_viz = viz_general_segments_img(
                    intermediates.patch_img,
                    submap.segments,
                    crop=crop,
                    px_per_m=px_per_m,
                    downsample_factor=params.aerial_viz_downsample,
                    line_width_m=params.aerial_viz_line_width_m,
                )
                viz_bytes = downsample_to_target_size(
                    sparse_general_viz, params.aerial_viz_target_size_kb
                )
                fname_sparse_general = viz_output_dir / f"{i}_{j}_sparse.jpg"
                with open(str(fname_sparse_general), "wb") as f:
                    f.write(viz_bytes)

    # ------------------------------------------------------------------
    # Ground segmentation with I/O
    # ------------------------------------------------------------------

    def run_ground(
        self,
        submaps: List[Submap],
        output_dir: Union[str, pathlib.Path],
    ) -> List[Submap]:
        result = self.algorithm.batch_ground_to_sparse_2d_submaps(
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

        params = self.algorithm.pipeline_params

        for k, submap_2d in enumerate(result.submaps):
            # Save submap
            fname_segment = segment_output_dir / f"{k}.pkl"
            submap_2d.save(fname_segment)

            # Save per-segment dense 2D points
            if result.intermediates is not None:
                intermediates = result.intermediates[k]
                dense_dir = segment_output_dir / f"{k}_dense"
                dense_dir.mkdir(parents=True, exist_ok=True)
                for aerial_seg in intermediates.aerial_segments:
                    dense_path = dense_dir / f"{aerial_seg.id}.pkl"
                    pts = aerial_seg.points
                    max_n = params.dense_points_max_n
                    if max_n is not None and len(pts) > max_n:
                        idx = np.round(np.linspace(0, len(pts) - 1, max_n)).astype(int)
                        pts = pts[idx]
                    with open(dense_path, "wb") as f:
                        pickle.dump(pts, f)

                # Save visualization
                fig, ax = viz_ground_segments(
                    intermediates.flattened_submap,
                    intermediates.aerial_segments,
                    intermediates.general_segments,
                    submap_2d.segments,
                    params.alpha_shape_alpha,
                    params.alpha_shape_grid_downsample,
                    params.alpha_shape_max_n_pts,
                    params.alpha_shape_ref_size_m,
                )
                fname_general = viz_output_dir / f"{k}.png"
                fig.savefig(fname_general, dpi=400)
                plt.close(fig)

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
            )
        if output_dir is not None:
            self._save_match_results(
                match_result,
                output_dir,
                aerial_img,
                aerial_submaps,
                ground_submaps,
                ground_dense_dir,
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
    ):
        output_dir = pathlib.Path(output_dir)
        viz_output_dir = output_dir / "viz"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        segments_output_dir = output_dir / "segments"
        segments_output_dir.mkdir(parents=True, exist_ok=True)

        params = self.algorithm.pipeline_params
        pixel_len_m = self.algorithm.aerial_segmenter.params.pixel_len_m
        px_per_m = 1.0 / pixel_len_m
        patch_size_px = int(params.aerial_img_patch_side_len_m * px_per_m)
        stride = int(patch_size_px * (1.0 - params.aerial_img_patch_overlap))

        def aerial_key_to_tuple(key):
            return tuple(int(x) for x in key.split("_"))

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

            details = match_result.match_details.get(ground_key, {})

            for aerial_key, single_result in details.items():
                i_a, j_a = aerial_key_to_tuple(aerial_key)
                result = single_result.pose_result

                # Match visualization
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

                # Get ground submap's full segments for dense viz
                ground_segments_all = None
                if ground_key in ground_submaps:
                    ground_segments_all = ground_submaps[ground_key].segments

                viz_cross_view_matches(
                    single_result.aerial_segs_processed,
                    single_result.ground_segs_processed,
                    result.associations,
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
                viz_bytes = downsample_to_target_size(
                    img_array, params.match_viz_target_size_kb
                )
                fname_viz = (
                    ground_sub_dir / f"ground_{ground_key}_aerial_{aerial_key}.jpg"
                )
                with open(fname_viz, "wb") as f:
                    f.write(viz_bytes)

                # Pose on aerial crop visualization
                if aerial_img is not None and not np.any(np.isnan(result.T_i_j)):
                    x1 = i_a * stride
                    y1 = j_a * stride
                    T_est_for_viz = (
                        result.T_i_j_hat
                        if not np.any(np.isnan(result.T_i_j_hat))
                        else None
                    )
                    viz_bytes = viz_pose_on_aerial_crop(
                        aerial_img,
                        (x1, y1),
                        result.T_i_j,
                        px_per_m,
                        patch_size_px,
                        T_est_for_viz,
                        params.aerial_viz_target_size_kb,
                    )
                    fname_pose = (
                        ground_sub_dir
                        / f"ground_{ground_key}_aerial_{aerial_key}_pose.jpg"
                    )
                    with open(fname_pose, "wb") as f:
                        f.write(viz_bytes)

                # Save matched segments
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
            results_matrix.plot(
                dist_thresh=params.match_viz_dist_thresh_m,
                angle_thresh_deg=params.match_viz_angle_thresh_deg,
            )
            fname_heatmap = viz_output_dir / f"ground_{ground_key}_all.png"
            plt.savefig(fname_heatmap, dpi=400)
            plt.close()

        self._write_results_summary(match_result.results, output_dir)

    @staticmethod
    def _write_results_summary(all_results, output_dir):
        # Use same thresholds as before (sensible defaults)
        dist_thresh = 5.0
        angle_thresh_deg = 10.0

        n_total = len(all_results)
        n_any_success = 0
        n_max_assoc_success = 0

        for ground_key, results_matrix in all_results.items():
            trans_errors = results_matrix.translation_error_m
            rot_errors = np.rad2deg(results_matrix.rotation_error_rad)
            num_assoc = results_matrix.num_associations

            # Metric 1: any crop below both thresholds
            success_mask = (trans_errors < dist_thresh) & (
                rot_errors < angle_thresh_deg
            )
            if np.any(success_mask):
                n_any_success += 1

            # Metric 2: crop(s) with max associations — majority correct
            valid_mask = num_assoc > 0
            if not np.any(valid_mask):
                continue
            max_assoc = np.nanmax(num_assoc[valid_mask])
            if max_assoc == 0:
                continue
            tied_mask = num_assoc == max_assoc
            tied_successes = np.sum(success_mask[tied_mask])
            tied_total = np.sum(tied_mask)
            if tied_successes > tied_total / 2:  # strict majority
                n_max_assoc_success += 1

        results_path = pathlib.Path(output_dir) / "results.txt"
        results_str = (
            f"Successful ground submap pose found: {n_any_success} / {n_total}\n"
            + "Successful ground submap pose using max number of "
            + f"associations: {n_max_assoc_success} / {n_total}\n"
        )
        print(results_str)
        with open(results_path, "w") as f:
            f.write(results_str)


# ---------------------------------------------------------------------------
# Top-level entry point (backward-compatible signature)
# ---------------------------------------------------------------------------


def cross_view_matching(
    params, output_dir, skip_aerial=False, skip_ground=False, skip_match=False
):
    pipeline_params = CrossViewMatchingParams.load(params)
    pipeline_params.output_directory = output_dir
    segment_match_params = SegmentMatchParams.load(params)
    segment_match_params.dim = 2

    try:
        pr_params = CrossViewPlaceRecognitionParams.load(params)
    except Exception:
        pr_params = None
    if pr_params is None and pipeline_params.matching_mode == "vpr":
        pr_params = CrossViewPlaceRecognitionParams()
    place_recognition = CrossViewPlaceRecognition(pr_params) if pr_params else None

    algorithm = CrossViewMatching(
        pipeline_params=pipeline_params,
        matcher=SegmentMatcher(segment_match_params),
        registerer=Registerer(RegisterParams.load(params)),
        aerial_segmenter=AerialSegmenter(AerialSegmenterParams.load(params)),
        ground_submap_params=SubmapParams.load(params),
        place_recognition=place_recognition,
    )
    pipeline = CrossViewMatchingPipeline(algorithm=algorithm)

    initial_aerial_segments = None
    initial_ground_segments = None

    # Load data
    data_params = CrossViewLocalizationDataParams.load(params)
    data = CrossViewLocalizationData.from_params(data_params)

    # Sync aerial segmenter pixel size with data
    algorithm.aerial_segmenter.params.pixel_len_m = data.aerial_img_scale

    # Set up output directories
    aerial_output_dir = os.path.join(output_dir, "aerial")
    ground_output_dir = os.path.join(output_dir, "ground")
    match_output_dir = os.path.join(output_dir, "match")

    # copy params to main output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(params):
        shutil.copy2(params, os.path.join(output_dir, os.path.basename(params)))
    else:
        shutil.copytree(params, output_dir, dirs_exist_ok=True)

    # Extract aerial segments
    if not skip_aerial:
        pipeline.run_aerial(data.aerial_img, aerial_output_dir, data.aerial_img_origin)

    # Extract ground segments
    if not skip_ground:
        ground_map = data.ground_map
        ground_submaps = algorithm.ground_map_to_submaps(ground_map)
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
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(expandvars_recursive(args.output))

    cross_view_matching(
        args.params, args.output, args.skip_aerial, args.skip_ground, args.skip_match
    )
