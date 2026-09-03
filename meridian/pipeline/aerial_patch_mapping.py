"""Standalone aerial patch segmentation and mapping pipeline.

Produces the same aerial submap pickles and visualizations as
cross_view_matching's aerial stage, without requiring ground maps or matching.

Usage:
    python3 -m meridian.pipeline.aerial_patch_mapping \
        -p path/to/params.yaml -o /path/to/output
"""

import argparse
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def aerial_patch_mapping(params, output_dir):
    """Run aerial patch segmentation and save submaps + visualizations.

    Args:
        params: Path to YAML params file (same format as cross_view_matching).
        output_dir: Output directory for segments/ and viz/ subdirectories.
    """
    # Load params
    data_params = CrossViewLocalizationDataParams.load(params)
    aerial_segmenter_params = AerialSegmenterParams.load(params)
    aerial_patch_params = AerialPatchParams.load(params)
    conversion_params = SegmentToPrimitiveConversionParams.load(params)
    viz_params = CrossViewVisualizationParams.load(params)

    try:
        pr_params = CrossViewPlaceRecognitionParams.load(params)
    except Exception:
        pr_params = None
    from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
    from meridian.params.cross_view_params import check_frame_descriptors_match

    descriptor_type = (
        check_frame_descriptors_match(params, pr_params.comparison)
        if pr_params is not None
        else None
    )
    place_recognition = (
        CrossViewPlaceRecognition(pr_params, descriptor_type) if pr_params else None
    )

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
                pixel_len_m,
                conversion_params.alpha_shape_alpha,
                conversion_params.alpha_shape_grid_downsample,
                conversion_params.alpha_shape_max_n_pts,
                conversion_params.alpha_shape_ref_size_m,
                max(1, round(viz_params.aerial_viz_pixel_size_m / pixel_len_m)),
                viz_params.aerial_viz_line_width_m,
                viz_params.aerial_viz_target_size_kb,
                px_per_m,
                i,
                j,
            )
            futures[future] = crop

        for future in as_completed(futures):
            for fname, viz_bytes in future.result():
                with open(str(viz_output_dir / fname), "wb") as f:
                    f.write(viz_bytes)

    print(f"Saved visualizations to {viz_output_dir}")

    # Save params and commit hash
    from meridian.utils import save_params, save_commit_hash

    all_params = [
        data_params,
        aerial_segmenter_params,
        aerial_patch_params,
        conversion_params,
        viz_params,
    ]
    if pr_params is not None:
        all_params.append(pr_params)
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
        required=True,
        help="Path to params YAML file.",
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

    if args.debug:
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )

    aerial_patch_mapping(args.params, args.output)
