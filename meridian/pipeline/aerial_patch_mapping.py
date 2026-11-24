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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from meridian.params import (
    AerialSegmenterParams,
    SegmentToPrimitiveConversionParams,
)
from meridian.params.cross_view_params import (
    CrossViewPlaceRecognitionParams,
    CrossViewVisualizationParams,
)
from meridian.params.params_checks import require_frame_descriptor
from meridian.params.data_params import CrossViewLocalizationDataParams
from meridian.params.segment_to_primitive_params import AerialPatchParams
from meridian.segmenter.aerial_segmenter import AerialSegmenter
from meridian.map2d.segment_to_primitive import SegmentToPrimitiveConverter
from meridian.map2d.aerial_patch_primitive_mapping import (
    AerialPatchPrimitiveMapping,
    AerialSegmentationResult,
)
from meridian.pipeline.data import CrossViewLocalizationData
from meridian.utils import save_commit_hash, save_params
from meridian.viz.cross_view_viz import (
    downsample_to_target_size,
    viz_aerial_segments,
    viz_general_segments_img,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Saving + visualizing aerial patches (shared with cross_view_matching)
# ---------------------------------------------------------------------------


def _aerial_viz_worker(
    patch_img,
    aerial_segments,
    general_segments,
    sparse_segments,
    crop,
    i,
    j,
    pixel_len_m,
    segment_border_type,
    segment_border_grid_downsample,
    segment_border_max_n_pts,
    concave_hull_ratio,
    alpha_shape_alpha,
    alpha_shape_ref_size_m,
    aerial_viz_downsample,
    aerial_viz_line_width_m,
    aerial_viz_target_size_kb,
    px_per_m,
):
    """Render 3 aerial viz images for one patch. Returns list of (filename, bytes)."""
    results = []

    aerial_viz = viz_aerial_segments(
        patch_img,
        aerial_segments,
        crop,
        pixel_len_m,
        alpha_shape_alpha,
        segment_border_grid_downsample,
        segment_border_max_n_pts,
        alpha_shape_ref_size_m,
        aerial_viz_downsample,
        line_width_m=aerial_viz_line_width_m,
        segment_border_type=segment_border_type,
        concave_hull_ratio=concave_hull_ratio,
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


class AerialPatchWriter:
    """Saves aerial submaps and renders their visualizations as patches finish.

    Pass as `on_patch_done` to `AerialPatchPrimitiveMapping.run` so each patch
    is written out (and its dense intermediates released) while segmentation of
    later patches continues, instead of holding every patch's intermediates
    until the end. Visualization runs in its own process pool with at most
    `max_pending` patches in flight. Use as a context manager (or call
    `close()`) to wait for the remaining visualizations.
    """

    def __init__(
        self,
        output_dir,
        pixel_len_m: float,
        conv_params: SegmentToPrimitiveConversionParams,
        viz_params: CrossViewVisualizationParams,
        max_workers: int,
        max_pending: int = None,
    ):
        output_dir = pathlib.Path(output_dir)
        self.segment_output_dir = output_dir / "segments"
        self.viz_output_dir = output_dir / "viz"
        self.segment_output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        self.pixel_len_m = pixel_len_m
        self.conv_params = conv_params
        self.viz_params = viz_params
        self.max_pending = max_pending or 2 * max(1, max_workers)
        self.n_saved = 0
        self._pending = set()
        self._executor = ProcessPoolExecutor(max_workers=max(1, max_workers))

    def __call__(self, crop, submap, intermediates):
        i, j = submap.id  # patch column / row index
        submap.save(self.segment_output_dir / f"{i}_{j}.pkl")
        self.n_saved += 1
        if intermediates is None:
            return
        c = self.conv_params
        self._pending.add(
            self._executor.submit(
                _aerial_viz_worker,
                intermediates.patch_img,
                intermediates.aerial_segments,
                intermediates.general_segments,
                submap.segments,
                crop,
                i,
                j,
                self.pixel_len_m,
                c.segment_border_type,
                c.segment_border_grid_downsample,
                c.segment_border_max_n_pts,
                c.concave_hull_ratio,
                c.alpha_shape_alpha,
                c.alpha_shape_ref_size_m,
                max(
                    1,
                    round(self.viz_params.aerial_viz_pixel_size_m / self.pixel_len_m),
                ),
                self.viz_params.aerial_viz_line_width_m,
                self.viz_params.aerial_viz_target_size_kb,
                1.0 / self.pixel_len_m,
            )
        )
        self._write_done(block=len(self._pending) >= self.max_pending)

    def _write_done(self, block: bool = False):
        if block:
            done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
        else:
            done = {f for f in self._pending if f.done()}
        for fut in done:
            self._pending.discard(fut)
            for fname, viz_bytes in fut.result():
                with open(str(self.viz_output_dir / fname), "wb") as f:
                    f.write(viz_bytes)

    def close(self):
        while self._pending:
            self._write_done(block=True)
        self._executor.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._executor.shutdown(wait=False, cancel_futures=True)


def segment_and_save_aerial(
    aerial_mapping: AerialPatchPrimitiveMapping,
    img,
    img_origin,
    output_dir,
    viz_params: CrossViewVisualizationParams,
) -> AerialSegmentationResult:
    """Run aerial patch mapping, saving each submap to `output_dir/segments` and
    its visualizations to `output_dir/viz` as soon as the patch finishes, so its
    dense intermediates are released right away (holding them all needs tens of
    GB on a large image)."""
    conv_params = aerial_mapping.converter.params
    with AerialPatchWriter(
        output_dir,
        pixel_len_m=aerial_mapping.aerial_segmenter.params.pixel_len_m,
        conv_params=conv_params,
        viz_params=viz_params,
        max_workers=conv_params.sparse_conversion_max_threads,
    ) as writer:
        result = aerial_mapping.run(
            img, img_origin, show_progress=True, on_patch_done=writer
        )
    logger.info(
        f"Saved {writer.n_saved} aerial submaps to {writer.segment_output_dir} "
        f"and visualizations to {writer.viz_output_dir}"
    )
    return result


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


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

    require_frame_descriptor(aerial_segmenter_params)

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
    )

    print("Running aerial patch segmentation...")
    output_dir = pathlib.Path(output_dir)
    result = segment_and_save_aerial(
        aerial_mapping,
        data.aerial_img,
        data.aerial_img_origin,
        output_dir,
        viz_params,
    )
    print(
        f"Saved {len(result.submaps)} aerial submaps + visualizations to {output_dir}"
    )
    t = result.timing
    print(
        f"Timing: {len(result.submaps)} patches in {t['total']:.0f} s; GPU "
        f"segmentation {t['segmentation']:.0f} s, paused waiting on conversion "
        f"{t['blocked']:.0f} s, conversion tail {t['conversion_tail']:.0f} s"
    )

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
