import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import robotdatapy as rdp
import time
from robotdatapy.data import PoseData, ImgData
import cv2 as cv
import os
import argparse
import trimesh
import pathlib
from tqdm import tqdm

from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.params.fastsam_params import FastSAMParams

from gen_seg_match.segment.segment_types import SegmentList
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.map3d.segments_from_img import (
    get_segments_with_occlusion,
    roman_segments_to_general_segments,
)
from gen_seg_match.params import (
    SegmentMatchParams,
    CrossViewLocalizationParams,
    CrossViewLocalizationDataParams,
    AerialSegmenterParams,
)
from gen_seg_match.viz.utils import color_from_seed
from gen_seg_match.viz.img_sparse_viz import img_sparse_viz
from gen_seg_match.pipeline.data import CrossViewLocalizationData
from gen_seg_match.utils import expandvars_recursive
from gen_seg_match.map2d.aerial_segmenter import AerialSegmenter
from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.segment.segment_types import SegmentPoint, SegmentLine

Crop = Tuple[int, int, int, int]


@dataclass
class CrossViewLocalization:
    pipeline_params: CrossViewLocalizationParams
    aerial_segmenter: AerialSegmenter
    matcher: SegmentMatcher

    def aerial_img_to_segments(self, img: np.ndarray, crop: Crop = None) -> SegmentList:
        segments = self.aerial_segmenter.run(img, crop=crop)
        for segment in segments:
            segment.get_alpha_shape(
                alpha=self.pipeline_params.aerial_alpha_shape_alpha,
                grid_downsample=self.pipeline_params.aerial_alpha_shape_grid_downsample,
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
                    )
                )
                continue
            # try:
            alpha_shape = segment.get_alpha_shape(
                grid_downsample=self.pipeline_params.aerial_alpha_shape_grid_downsample,
                alpha=self.pipeline_params.aerial_alpha_shape_alpha,
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
                        )
                    )
        result = SegmentList(center_points + lines)
        result.reindex()
        return result

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
            segment_output_dir = output_dir / "fine_segments"
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
                results[crop] = general_segments

                # ------------------------------------------------------
                # 3. Visualization and store segments (if output_dir provided)
                # ------------------------------------------------------
                if output_dir is not None:
                    # -------- Save aerial segments --------
                    fname_segment = segment_output_dir / f"{i}_{j}.pkl"
                    general_segments.save(fname_segment)

                    # -------- Raw AerialSegments overlay --------
                    aerial_viz = self._viz_aerial_segments(
                        patch_img, aerial_segments, crop
                    )
                    fname_aerial = viz_output_dir / f"{i}_{j}_segments.png"
                    cv.imwrite(str(fname_aerial), aerial_viz)

                    # -------- GeneralSegments overlay --------
                    general_viz = self._viz_general_segments(
                        patch_img, general_segments, crop=crop
                    )

                    fname_general = viz_output_dir / f"{i}_{j}_fine.png"
                    cv.imwrite(str(fname_general), general_viz)

        return results

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
        print(img_origin_m)
        for seg in segments:
            alpha_shape_px = seg.get_alpha_shape_pixels(
                img_pixel_scale=self.aerial_segmenter.params.pixel_len_m,
                grid_downsample=self.pipeline_params.aerial_alpha_shape_grid_downsample,
                alpha=self.pipeline_params.aerial_alpha_shape_alpha,
                img_origin_m=img_origin_m,
            )
            if alpha_shape_px is None:
                continue
            # TODO: add some viz params
            cv.polylines(aerial_viz, [alpha_shape_px], True, seg.viz_color[::-1], 20)
        return aerial_viz

    def _viz_general_segments(
        self, img: np.ndarray, segments: SegmentList, crop: Crop
    ) -> np.ndarray:
        general_viz = img.copy()
        x1, y1, x2, y2 = crop
        px_per_m = 1.0 / self.aerial_segmenter.params.pixel_len_m

        # draw points
        for seg in segments:
            if isinstance(seg, SegmentPoint):
                p = seg.get_point()
                cv.circle(
                    general_viz,
                    (int(p[0] * px_per_m - x1), int(p[1] * px_per_m - y1)),
                    10,  # TODO: add some viz params
                    seg.color_from_id(order="bgr"),
                    10,  # TODO: add some viz params
                )

        # draw lines
        for seg in segments:
            if isinstance(seg, SegmentLine):
                p0 = seg.endpoints[0]
                p1 = seg.endpoints[1]
                cv.line(
                    general_viz,
                    (int(p0[0] * px_per_m - x1), int(p0[1] * px_per_m - y1)),
                    (int(p1[0] * px_per_m - x1), int(p1[1] * px_per_m - y1)),
                    seg.color_from_id(order="bgr"),
                    20,  # TODO: add some viz params
                )
        return general_viz


def cross_view_localization(params, output_dir):
    pipeline_params = CrossViewLocalizationParams.load(params)
    pipeline_params.output_directory = output_dir
    runner = CrossViewLocalization(
        pipeline_params=pipeline_params,
        matcher=SegmentMatcher(SegmentMatchParams.load(params)),
        aerial_segmenter=AerialSegmenter(AerialSegmenterParams.load(params)),
    )

    # Load data
    data_params = CrossViewLocalizationDataParams.load(params)
    data = CrossViewLocalizationData.from_params(data_params)

    # Set up output directories
    aerial_output_dir = os.path.join(output_dir, "aerial")
    ground_output_dir = os.path.join(output_dir, "ground")
    match_output_dir = os.path.join(output_dir, "match")

    # Extract aerial segments
    initial_aerial_segments = runner.batch_aerial_img_to_segments(
        data.aerial_img, aerial_output_dir
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
        + "aerial_segmenter, segment_match",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    # TODO: do I need this?
    # parser.add_argument(
    #     "-r", "--runs", type=str, nargs=2, required=True, help="Run names."
    # )
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(expandvars_recursive(args.output))

    cross_view_localization(args.params, args.output)
