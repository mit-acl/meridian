import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import cv2 as cv
import os
import argparse
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.params.fastsam_params import FastSAMParams
from roman.map.map import ROMANMap

from gen_seg_match.segment.segment_types import SegmentList
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.params import (
    SegmentMatchParams,
    CrossViewLocalizationParams,
    CrossViewLocalizationDataParams,
    AerialSegmenterParams,
    SubmapParams,
    GroundSegmenterParams,
)
from gen_seg_match.viz.utils import color_from_seed
from gen_seg_match.viz.img_sparse_viz import img_sparse_viz
from gen_seg_match.pipeline.data import CrossViewLocalizationData
from gen_seg_match.utils import expandvars_recursive
from gen_seg_match.map2d.aerial_segmenter import AerialSegmenter
from gen_seg_match.map2d.ground_segmenter import GroundSegmenter
from gen_seg_match.map2d.map_processing import clean_up_line_map
from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.segment.segment_types import SegmentPoint, SegmentLine
from gen_seg_match.map3d.submap import (
    Submap,
    submaps_from_roman_map,
    RomanConversionParams,
)
from gen_seg_match.viz.cross_view_viz import viz_cross_view_matches

Crop = Tuple[int, int, int, int]


@dataclass
class CrossViewLocalization:
    pipeline_params: CrossViewLocalizationParams
    aerial_segmenter: AerialSegmenter
    matcher: SegmentMatcher
    ground_submap_params: SubmapParams = None
    ground_segmenter: GroundSegmenter = None

    def __post_init__(self):
        if self.ground_segmenter is None:
            self.ground_segmenter = GroundSegmenter(GroundSegmenterParams())
        if self.ground_submap_params is None:
            self.ground_submap_params = SubmapParams()

    def aerial_img_to_segments(self, img: np.ndarray, crop: Crop = None) -> SegmentList:
        segments = self.aerial_segmenter.run(img, crop=crop)
        for segment in segments:
            segment.get_alpha_shape(
                alpha=self.pipeline_params.alpha_shape_alpha,
                grid_downsample=self.pipeline_params.alpha_shape_grid_downsample,
            )
        return segments

    def aerial_segments_to_general_segments(
        self, segments: List[AerialSegment], crop: Crop = None
    ) -> SegmentList:
        # precomputation
        if crop is not None:
            x1, y1, x2, y2 = crop
            pixel_len_m = self.aerial_segmenter.params.pixel_len_m
            min_dist_m = self.pipeline_params.aerial_min_dist_to_border_m
            x1_border = x1 * pixel_len_m + min_dist_m
            y1_border = y1 * pixel_len_m + min_dist_m
            x2_border = x2 * pixel_len_m - min_dist_m
            y2_border = y2 * pixel_len_m - min_dist_m
        else:
            # No crop --> no border restrictions
            x1_border = -np.inf
            y1_border = -np.inf
            x2_border = np.inf
            y2_border = np.inf

        lines = []
        center_points = []
        for j, segment in enumerate(segments):
            if segment.area < self.pipeline_params.point_max_area_m_sq:
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
            # try:
            alpha_shape = segment.get_alpha_shape(
                grid_downsample=self.pipeline_params.alpha_shape_grid_downsample,
                alpha=self.pipeline_params.alpha_shape_alpha,
            )
            if alpha_shape is None:
                continue
            # print(alpha_shape)
            for i, pt0 in enumerate(alpha_shape):
                pt1 = alpha_shape[i + 1 if i + 1 < len(alpha_shape) else 0]
                keep = (
                    np.linalg.norm(pt1 - pt0) > self.pipeline_params.line_min_length_m
                )
                keep &= pt0[0] >= x1_border or pt1[0] >= x1_border
                keep &= pt0[0] <= x2_border or pt1[0] <= x2_border
                keep &= pt0[1] >= y1_border or pt1[1] >= y1_border
                keep &= pt0[1] <= y2_border or pt1[1] <= y2_border
                if keep:
                    lines.append(
                        SegmentLine.from_endpoints(
                            j,
                            pt0,
                            pt1,
                            cos_feature=segment.semantic_descriptor,
                            first_seen=segment.first_seen,
                            last_seen=segment.last_seen,
                            history=[segment.id],
                        )
                    )
        result = SegmentList(center_points + lines)
        result.reindex()
        return result

    def ground_map_to_submaps(self, ground_map: ROMANMap) -> List[Submap]:
        conversion_params = RomanConversionParams(
            copy_dense_points=True, force_points_only=True
        )
        submaps = submaps_from_roman_map(
            ground_map, self.ground_submap_params, conversion_params
        )
        return submaps

    def load_segments_from_dir(
        self,
        segment_dir: Union[str, pathlib.Path],
    ) -> Dict[str, SegmentList]:
        """
        Loads segments from a directory.

        Args:
            segment_dir (Union[str, pathlib.Path]): Directory containing segment files.

        Returns:
            Dict[str, SegmentList]: List of extracted segments for each submap
        """
        segment_dir = pathlib.Path(segment_dir)
        segment_files = list(segment_dir.glob("*.pkl"))
        segment_files.sort()
        results = {}
        for segment_file in segment_files:
            segments = SegmentList.load(segment_file)
            results[segment_file.stem] = segments
        return results

    def batch_aerial_img_to_segments(
        self, img: np.ndarray, output_dir: Union[str, pathlib.Path] = None
    ) -> Dict[Crop, SegmentList]:
        """
        Batch process of an aerial image. Splits the image into patches,
            processes each patch to extract fine-grained segments, and combines the results.
            If output_dir is provided, saves visualizations of the segments.

        Args:
            img (np.ndarray): Aerial image

        Returns:
            Dict[Crop, SegmentList]: Aerial image crop to extracted fine-grained segments
        """

        # iterate over the images with patch size and overlap
        # defined from pipeline_params
        h, w = img.shape[:2]
        px_per_m = 1.0 / self.aerial_segmenter.params.pixel_len_m

        patch_size_m = self.pipeline_params.aerial_img_patch_side_len_m
        patch_size_px = int(patch_size_m * px_per_m)

        overlap = self.pipeline_params.aerial_img_patch_overlap
        stride = int(patch_size_px * (1.0 - overlap))

        if stride <= 0:
            raise ValueError("Patch overlap too large; stride becomes non-positive.")

        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            viz_output_dir = output_dir / "viz"
            segment_output_dir = output_dir / "segments"
            viz_output_dir.mkdir(parents=True, exist_ok=True)
            segment_output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # ------------------------------------------------------
        # Patch iteration
        # ------------------------------------------------------
        for j, y1 in enumerate(tqdm(range(0, h - patch_size_px + 1, stride))):
            for i, x1 in enumerate(range(0, w - patch_size_px + 1, stride)):
                x2 = x1 + patch_size_px
                y2 = y1 + patch_size_px
                crop = (x1, y1, x2, y2)

                patch_img = img[y1:y2, x1:x2].copy()

                # ------------------------------------------------------
                # 1. Run aerial segmentation (alpha shape computed inside)
                # ------------------------------------------------------
                aerial_segments = self.aerial_img_to_segments(img, crop=crop)

                # ------------------------------------------------------
                # 2. Convert to general segments
                # ------------------------------------------------------
                general_segments = self.aerial_segments_to_general_segments(
                    aerial_segments, crop=crop
                )
                sparse_general_segments = (
                    general_segments.get_points()
                    + clean_up_line_map(
                        general_segments.get_lines(),
                        angle_tol=self.pipeline_params.line_merge_ang_thresh_rad,
                        dist_tol=self.pipeline_params.line_merge_dist_thresh_m,
                    )[0]
                )
                sparse_general_segments.reindex()

                results[crop] = sparse_general_segments

                # ------------------------------------------------------
                # 3. Visualization and store segments (if output_dir provided)
                # ------------------------------------------------------
                if output_dir is not None:
                    # -------- Save aerial segments --------
                    fname_segment = segment_output_dir / f"{i}_{j}.pkl"
                    sparse_general_segments.save(fname_segment)

                    # -------- Raw AerialSegments overlay --------
                    aerial_viz = self._viz_aerial_segments(
                        patch_img, aerial_segments, crop
                    )
                    fname_aerial = viz_output_dir / f"{i}_{j}_segments.png"
                    cv.imwrite(str(fname_aerial), aerial_viz)

                    # -------- GeneralSegments overlay --------
                    general_viz = self._viz_general_segments_img(
                        patch_img, general_segments, crop=crop
                    )
                    fname_general = viz_output_dir / f"{i}_{j}_fine.png"
                    cv.imwrite(str(fname_general), general_viz)

                    # --------- Sparse GeneralSegments overlay --------
                    sparse_general_viz = self._viz_general_segments_img(
                        patch_img, sparse_general_segments, crop=crop
                    )
                    fname_sparse_general = viz_output_dir / f"{i}_{j}_sparse.png"
                    cv.imwrite(str(fname_sparse_general), sparse_general_viz)

        return results

    def batch_ground_submaps_to_segments(
        self, submaps: List[Submap], output_dir: Union[str, pathlib.Path] = None
    ) -> List[SegmentList]:
        """
        Batch process of ground submaps. For each submap, creates a 2D aerial segments
            and if an output_dir is provided, saves visualizations of the segments.

        Args:
            submaps (List[Submap]): List of ground submaps
            output_dir (Union[str, pathlib.Path], optional): Directory to save visualizations. Defaults to None.

        Returns:
            List[SegmentList]: List of extracted fine-grained segments for each submap
        """
        results = []

        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            viz_output_dir = output_dir / "viz"
            segment_output_dir = output_dir / "segments"
            viz_output_dir.mkdir(parents=True, exist_ok=True)
            segment_output_dir.mkdir(parents=True, exist_ok=True)

        for k, submap in enumerate(tqdm(submaps)):
            # first transform the submap from submap frame to 3D odometry for now
            submap.segments.transform(submap.pose_flu)

            flattened_submap = self.ground_segmenter.flatten_3d_submap(submap)
            aerial_segments = self.ground_segmenter.submap_2d_to_aerial(
                flattened_submap
            )
            aerial_segments = [
                seg
                for seg in aerial_segments
                if seg.get_alpha_shape(
                    alpha=self.pipeline_params.alpha_shape_alpha,
                    grid_downsample=self.pipeline_params.alpha_shape_grid_downsample,
                )
                is not None
            ]
            general_segments = self.aerial_segments_to_general_segments(aerial_segments)
            sparse_general_segments = (
                general_segments.get_points()
                + clean_up_line_map(
                    general_segments.get_lines(),
                    angle_tol=self.pipeline_params.line_merge_ang_thresh_rad,
                    dist_tol=self.pipeline_params.line_merge_dist_thresh_m,
                )[0]
            )
            sparse_general_segments.reindex()
            for seg in sparse_general_segments:
                history_heights = [
                    submap.segments.get_segment_from_id(id_hist).point.item(2)
                    for id_hist in seg.history
                ]
                seg.height = np.mean(history_heights)
            results.append(sparse_general_segments)

            if output_dir is not None:
                # -------- Save aerial segments --------
                fname_segment = segment_output_dir / f"{k}.pkl"
                sparse_general_segments.save(fname_segment)

                # -------- GeneralSegments overlay --------
                fig, ax = self._viz_ground_segments(
                    flattened_submap,
                    aerial_segments,
                    general_segments,
                    sparse_general_segments,
                )
                fname_general = viz_output_dir / f"{k}.png"
                fig.savefig(fname_general, dpi=400)

        return results

    def batch_cross_view_match(
        self,
        aerial_segments: Dict[str, SegmentList],
        ground_segments: Dict[str, SegmentList],
        output_dir: Union[str, pathlib.Path] = None,
    ) -> None:
        """
        Batch process of cross-view segment matching. For each aerial image crop and ground submap,
            matches segments and if an output_dir is provided, saves visualizations of the matches.

        Args:
            aerial_segments (Dict[str, SegmentList]): Aerial image name to extracted segments
            ground_segments (Dict[str, SegmentList]): Ground image name to extracted segments
            output_dir (Union[str, pathlib.Path], optional): Directory to save visualizations. Defaults to None.
        """
        assert output_dir is not None, (
            "Output directory must be provided for match visualization."
        )

        output_dir = pathlib.Path(output_dir)
        viz_output_dir = output_dir / "viz"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        segments_output_dir = output_dir / "segments"
        segments_output_dir.mkdir(parents=True, exist_ok=True)

        aerial_key_to_tuple = lambda key: tuple(int(x) for x in key.split("_"))
        aerial_x_max = np.max(
            [aerial_key_to_tuple(key)[0] for key in aerial_segments.keys()]
        )
        aerial_y_max = np.max(
            [aerial_key_to_tuple(key)[1] for key in aerial_segments.keys()]
        )

        # minor processing on segments
        # TODO: put to 3d (dim = 3) and filter by length
        aerial_segments_2d = {}
        ground_segments_2d = {}
        for segments_2d_dict, original_segments_dict in [
            (aerial_segments_2d, aerial_segments),
            (ground_segments_2d, ground_segments),
        ]:
            for key, segments in original_segments_dict.items():
                segments_2d = segments.to_dim(2)
                filtered_lines = [
                    line
                    for line in segments_2d.get_lines()
                    if line.get_length() >= self.pipeline_params.match_min_len_m
                ]
                segments_2d = segments_2d.get_points() + SegmentList(filtered_lines)
                segments_2d_dict[key] = segments_2d

        # make sure to transfer height (TODO: figure out how to handle this cleanly)
        for ground_key in ground_segments_2d.keys():
            for segment in ground_segments_2d[ground_key]:
                segment.height = (
                    ground_segments[ground_key].get_segment_from_id(segment.id).height
                )

        # iterate over all aerial crops and ground submaps
        for ground_key, ground_segs_i in tqdm(ground_segments_2d.items()):
            num_associations = np.zeros((aerial_x_max + 1, aerial_y_max + 1), dtype=int)
            ground_sub_dir = viz_output_dir / f"ground_{ground_key}"
            ground_sub_dir.mkdir(parents=True, exist_ok=True)
            for aerial_key, aerial_segs_j in aerial_segments_2d.items():
                matches = self.matcher.match(
                    ground_segs_i,
                    aerial_segs_j,
                )

                # -------- Match visualization --------
                viz_cross_view_matches(aerial_segs_j, ground_segs_i, matches)
                fname_viz = (
                    ground_sub_dir / f"ground_{ground_key}_aerial_{aerial_key}.png"
                )
                fig = plt.gcf()
                fig.savefig(fname_viz, dpi=400)
                plt.close(fig)
                num_associations[*aerial_key_to_tuple(aerial_key)] = len(matches)

                # -------- Save matches --------
                matched_ground = SegmentList(
                    [ground_segs_i.get_segment_from_id(g_id) for g_id, _ in matches]
                )
                matched_aerial = SegmentList(
                    [aerial_segs_j.get_segment_from_id(a_id) for _, a_id in matches]
                )
                fname_matches = (
                    segments_output_dir / f"ground_{ground_key}_aerial_{aerial_key}.pkl"
                )
                with open(fname_matches, "wb") as f:
                    pickle.dump([matched_ground, matched_aerial], f)

            # Save number of associations heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(num_associations.T)
            plt.colorbar(label="Number of Matches")
            plt.xlabel("Aerial Crop X Index")
            plt.ylabel("Aerial Crop Y Index")
            plt.title(f"Number of Matches for Ground Submap {ground_key}")
            fname_heatmap = viz_output_dir / f"ground_{ground_key}_all.png"
            plt.savefig(fname_heatmap, dpi=400)
            plt.close()

    # TODO: all of these visualizations should probably be moved to the viz module
    def _viz_aerial_segments(
        self,
        img: np.ndarray,
        segments: List[AerialSegment],
        crop: Crop = None,
    ) -> np.ndarray:
        aerial_viz = img.copy()
        img_origin_m = (
            (0.0, 0.0)
            if crop is None
            else (
                crop[0] * self.aerial_segmenter.params.pixel_len_m,
                crop[1] * self.aerial_segmenter.params.pixel_len_m,
            )
        )
        for seg in segments:
            alpha_shape_px = seg.get_alpha_shape_pixels(
                img_pixel_scale=self.aerial_segmenter.params.pixel_len_m,
                grid_downsample=self.pipeline_params.alpha_shape_grid_downsample,
                alpha=self.pipeline_params.alpha_shape_alpha,
                img_origin_m=img_origin_m,
            )
            if alpha_shape_px is None:
                continue
            # TODO: add some viz params
            cv.polylines(aerial_viz, [alpha_shape_px], True, seg.viz_color[::-1], 20)

        # downsample for viz
        return self._downsample_aerial_viz(aerial_viz)

    def _viz_general_segments_img(
        self, img: np.ndarray, segments: SegmentList, crop: Crop
    ) -> np.ndarray:
        general_viz = img.copy()
        x1, y1, x2, y2 = crop
        px_per_m = 1.0 / self.aerial_segmenter.params.pixel_len_m

        # draw points
        for seg in segments.get_points():
            p = seg.get_point()
            cv.circle(
                general_viz,
                (int(p[0] * px_per_m - x1), int(p[1] * px_per_m - y1)),
                10,  # TODO: add some viz params
                seg.color_from_id(order="bgr"),
                10,  # TODO: add some viz params
            )

        # draw lines
        for seg in segments.get_lines():
            p0 = seg.endpoints[0]
            p1 = seg.endpoints[1]
            cv.line(
                general_viz,
                (int(p0[0] * px_per_m - x1), int(p0[1] * px_per_m - y1)),
                (int(p1[0] * px_per_m - x1), int(p1[1] * px_per_m - y1)),
                seg.color_from_id(order="bgr"),
                20,  # TODO: add some viz params
            )
        return self._downsample_aerial_viz(general_viz)

    def _viz_ground_segments(
        self,
        flattened_submap: Submap,
        aerial_segments: List[AerialSegment],
        general_segments: SegmentList,
        sparse_general_segments: SegmentList,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # Plot just segment points
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        for seg in flattened_submap.segments:
            ax[0, 0].plot(
                seg.dense_points[:, 0],
                seg.dense_points[:, 1],
                ".",
                linewidth=1.0,
                alpha=0.5,
                color=seg.color_from_id(num_type=float),
            )
        ax[0, 0].set_aspect("equal")

        # Plot aerial segments (alpha shapes)
        for seg in aerial_segments:
            ax[0, 1].plot(
                seg.get_alpha_shape(self.pipeline_params.alpha_shape_alpha)[:, 0],
                seg.get_alpha_shape(self.pipeline_params.alpha_shape_alpha)[:, 1],
                color=seg.color_from_id(num_type=float),
                linewidth=2,
            )
        ax[0, 1].set_aspect("equal")

        # Plot general segments (points and lines)
        self._viz_general_segments_plt(ax[1, 0], general_segments)
        self._viz_general_segments_plt(ax[1, 1], sparse_general_segments)

        xlim = ax[0, 0].get_xlim()
        ylim = ax[0, 0].get_ylim()
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlim(xlim)
                ax[i, j].set_ylim(ylim)
        ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        if ratio > 1:
            fig.set_size_inches(10, 10 / ratio)
        else:
            fig.set_size_inches(10 * ratio, 10)

        return fig, ax

    def _viz_general_segments_plt(
        self, ax: plt.Axes, general_segments: SegmentList
    ) -> plt.Axes:
        for seg in general_segments.get_points():
            p = seg.get_point()
            ax.plot(
                p[0],
                p[1],
                "o",
                markersize=4,
                color=seg.color_from_id(num_type=float),
            )

        for seg in general_segments.get_lines():
            p0 = seg.endpoints[0]
            p1 = seg.endpoints[1]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                "-",
                linewidth=2,
                color=seg.color_from_id(num_type=float),
            )
        ax.set_aspect("equal")
        return ax

    def _downsample_aerial_viz(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(
            img,
            (
                img.shape[1] // self.pipeline_params.aerial_viz_downsample,
                img.shape[0] // self.pipeline_params.aerial_viz_downsample,
            ),
            interpolation=cv.INTER_AREA,
        )


def cross_view_localization(
    params, output_dir, skip_aerial=False, skip_ground=False, skip_match=False
):
    pipeline_params = CrossViewLocalizationParams.load(params)
    pipeline_params.output_directory = output_dir
    segment_match_params = SegmentMatchParams.load(params)
    segment_match_params.dim = 2
    runner = CrossViewLocalization(
        pipeline_params=pipeline_params,
        matcher=SegmentMatcher(segment_match_params),
        aerial_segmenter=AerialSegmenter(AerialSegmenterParams.load(params)),
        ground_submap_params=SubmapParams.load(params),
    )
    initial_aerial_segments = None
    initial_ground_segments = None

    # Load data
    data_params = CrossViewLocalizationDataParams.load(params)
    data = CrossViewLocalizationData.from_params(data_params)

    # Set up output directories
    aerial_output_dir = os.path.join(output_dir, "aerial")
    ground_output_dir = os.path.join(output_dir, "ground")
    match_output_dir = os.path.join(output_dir, "match")

    # Extract aerial segments
    if not skip_aerial:
        initial_aerial_segments = runner.batch_aerial_img_to_segments(
            data.aerial_img, aerial_output_dir
        )
        initial_aerial_segments = {
            "{i}_{j}": segs for (i, j), segs in initial_aerial_segments.items()
        }

    # Extract ground segments
    if not skip_ground:
        ground_map = data.ground_map
        ground_submaps = runner.ground_map_to_submaps(ground_map)
        initial_ground_segments = runner.batch_ground_submaps_to_segments(
            ground_submaps, ground_output_dir
        )
        initial_ground_segments = {
            str(k): segs for k, segs in enumerate(initial_ground_segments)
        }

    if not skip_match:
        if initial_aerial_segments is None:
            initial_aerial_segments = runner.load_segments_from_dir(
                os.path.join(aerial_output_dir, "segments")
            )
        if initial_ground_segments is None:
            initial_ground_segments = runner.load_segments_from_dir(
                os.path.join(ground_output_dir, "segments")
            )
        runner.batch_cross_view_match(
            initial_aerial_segments,
            initial_ground_segments,
            match_output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to params directory or file. "
        + "Required params: cross_view_localization, cross_view_localization_data, "
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
    # TODO: do I need this?
    # parser.add_argument(
    #     "-r", "--runs", type=str, nargs=2, required=True, help="Run names."
    # )
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(expandvars_recursive(args.output))

    cross_view_localization(
        args.params, args.output, args.skip_aerial, args.skip_ground, args.skip_match
    )
