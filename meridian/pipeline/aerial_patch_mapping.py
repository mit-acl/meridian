"""Standalone aerial patch segmentation and mapping pipeline.

Produces the same aerial submap pickles and visualizations as
cross_view_matching's aerial stage, without requiring ground maps or matching.

Usage:
    python3 -m meridian.pipeline.aerial_patch_mapping \
        -p path/to/params.yaml -o /path/to/output

    # default params, GeoTIFF input only
    python3 -m meridian.pipeline.aerial_patch_mapping \
        -a path/to/aerial.tiff [-s DOWNSAMPLE] -o /path/to/output
"""

import argparse
import logging
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
from meridian.params import (
    AerialSegmenterParams,
    SegmentToPrimitiveConversionParams,
)
from meridian.params.cross_view_params import (
    CrossViewPlaceRecognitionParams,
    CrossViewVisualizationParams,
)
from meridian.params.data_params import CrossViewLocalizationDataParams
from meridian.params.segment_to_primitive_params import AerialPatchParams
from meridian.segmenter.aerial_segmenter import AerialSegmenter
from meridian.map2d.segment_to_primitive import SegmentToPrimitiveConverter
from meridian.map2d.aerial_patch_primitive_mapping import (
    AerialPatchPrimitiveMapping,
)
from meridian.pipeline.cross_view_matching import _aerial_viz_worker
from meridian.pipeline.data import CrossViewLocalizationData
from meridian.utils import save_commit_hash, save_params
from meridian.vpr.vpr import check_frame_descriptors_match


def aerial_patch_mapping(
    params, output_dir, aerial_img_path=None, downsample_factor=None
):
    """Run aerial patch segmentation and save submaps + visualizations.

    Args:
        params: Path to YAML params file (same format as cross_view_matching), or
            None to use default params (requires aerial_img_path).
        output_dir: Output directory for segments/ and viz/ subdirectories.
        aerial_img_path: Aerial image path; overrides the params file if both are set.
        downsample_factor: Aerial segmenter downsample factor; None keeps the
            params-file / default value.
    """
    if params is None and aerial_img_path is None:
        raise ValueError("Provide a params file and/or an aerial image path.")

    # Load params (defaults when no params file is given)
    if params is not None:
        data_params = CrossViewLocalizationDataParams.load(params)
        aerial_segmenter_params = AerialSegmenterParams.load(params)
        aerial_patch_params = AerialPatchParams.load(params)
        conversion_params = SegmentToPrimitiveConversionParams.load(params)
        viz_params = CrossViewVisualizationParams.load(params)
        # A missing section loads defaults; real config errors should raise
        # rather than silently produce submaps without descriptors.
        pr_params = CrossViewPlaceRecognitionParams.load(params)
    else:
        data_params = CrossViewLocalizationDataParams(aerial_img_path=aerial_img_path)
        aerial_segmenter_params = AerialSegmenterParams()
        aerial_patch_params = AerialPatchParams()
        conversion_params = SegmentToPrimitiveConversionParams()
        viz_params = CrossViewVisualizationParams()
        pr_params = CrossViewPlaceRecognitionParams()

    # Command-line overrides
    if params is not None and aerial_img_path is not None:
        yellow, reset = "\033[1;33m", "\033[0m"
        bar = "=" * 80
        print(
            f"{yellow}{bar}\n"
            "WARNING: aerial image set by both --aerial-img and the params file.\n"
            f"  Loading from --aerial-img:       {aerial_img_path}\n"
            f"  Ignoring params `aerial_img_path`: {data_params.aerial_img_path}\n"
            f"{bar}{reset}",
            file=sys.stderr,
            flush=True,
        )
        data_params.aerial_img_path = aerial_img_path
    if downsample_factor is not None:
        aerial_segmenter_params.downsample_factor = downsample_factor

    # Model that makes the aerial descriptors: checked against the ground
    # segmenter's when a params file configures either; otherwise the aerial
    # segmenter's (possibly default) frame_descriptor.
    descriptor_type = (
        check_frame_descriptors_match(params, pr_params.comparison)
        if params is not None
        else None
    )
    if descriptor_type is None:
        descriptor_type = aerial_segmenter_params.frame_descriptor
    place_recognition = CrossViewPlaceRecognition(pr_params, descriptor_type)

    # Load aerial image
    print("Loading aerial image...")
    data = CrossViewLocalizationData.from_params(data_params)
    aerial_segmenter_params.pixel_len_m = data.aerial_img_scale

    # Build pipeline
    aerial_segmenter = AerialSegmenter(aerial_segmenter_params)
    converter = SegmentToPrimitiveConverter(conversion_params)
    aerial_mapping = AerialPatchPrimitiveMapping(
        patch_params=aerial_patch_params,
        converter=converter,
        aerial_segmenter=aerial_segmenter,
        place_recognition=place_recognition,
    )

    # Run segmentation
    print("Running aerial patch segmentation...")
    result = aerial_mapping.run(
        data.aerial_img,
        data.aerial_img_origin,
        return_intermediates=True,
        show_progress=True,
    )

    # Save results
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir = output_dir / "viz"
    segment_output_dir = output_dir / "segments"
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    segment_output_dir.mkdir(parents=True, exist_ok=True)

    pixel_len_m = aerial_segmenter_params.pixel_len_m
    px_per_m = 1.0 / pixel_len_m
    patch_size_px = int(aerial_patch_params.aerial_img_patch_side_len_m * px_per_m)
    stride = int(patch_size_px * (1.0 - aerial_patch_params.aerial_img_patch_overlap))

    # Save submaps
    crop_ij = {}
    for crop, submap in result.submaps.items():
        x1, y1, x2, y2 = crop
        i = x1 // stride
        j = y1 // stride
        crop_ij[crop] = (i, j)
        fname = segment_output_dir / f"{i}_{j}.pkl"
        submap.save(fname)

    print(f"Saved {len(result.submaps)} aerial submaps to {segment_output_dir}")

    # Parallel visualization
    max_workers = conversion_params.sparse_conversion_max_threads
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
                i,
                j,
                pixel_len_m,
                conversion_params.segment_border_type,
                conversion_params.segment_border_grid_downsample,
                conversion_params.segment_border_max_n_pts,
                conversion_params.concave_hull_ratio,
                conversion_params.alpha_shape_alpha,
                conversion_params.alpha_shape_ref_size_m,
                max(1, round(viz_params.aerial_viz_pixel_size_m / pixel_len_m)),
                viz_params.aerial_viz_line_width_m,
                viz_params.aerial_viz_target_size_kb,
                px_per_m,
            )
            futures[future] = crop

        for future in as_completed(futures):
            for fname, viz_bytes in future.result():
                with open(str(viz_output_dir / fname), "wb") as f:
                    f.write(viz_bytes)

    print(f"Saved visualizations to {viz_output_dir}")

    # Save params and commit hash
    all_params = [
        data_params,
        aerial_segmenter_params,
        aerial_patch_params,
        conversion_params,
        viz_params,
        pr_params,
    ]
    save_params(str(output_dir), *all_params)
    save_commit_hash(str(output_dir))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aerial patch segmentation and mapping pipeline."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Path to params YAML file. If omitted, default params are used "
        "(requires --aerial-img).",
    )
    parser.add_argument(
        "-a",
        "--aerial-img",
        type=str,
        default=None,
        help="Aerial image path (GeoTIFF when no params file). Overrides the "
        "params file's `aerial_img_path` if both are given.",
    )
    parser.add_argument(
        "-s",
        "--downsample",
        type=int,
        default=None,
        help="Aerial segmenter downsample factor (default: params file / "
        "AerialSegmenterParams default).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args()
    if args.params is None and args.aerial_img is None:
        parser.error("one of -p/--params or -a/--aerial-img is required")

    if args.debug:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )

    aerial_patch_mapping(
        args.params,
        args.output,
        aerial_img_path=args.aerial_img,
        downsample_factor=args.downsample,
    )
