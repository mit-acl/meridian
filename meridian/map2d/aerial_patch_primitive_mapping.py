import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from meridian.segmenter.aerial_segmenter import AerialSegmenter
from meridian.map2d.segment_to_primitive import SegmentToPrimitiveConverter
from meridian.map3d.submap import FrameType, Submap
from meridian.params.segment_to_primitive_params import AerialPatchParams
from meridian.map2d.segment2d import Segment2D
from meridian.primitive.primitive_list import PrimitiveList

logger = logging.getLogger(__name__)

Crop = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AerialPatchIntermediates:
    patch_img: np.ndarray = None
    aerial_segments: List[Segment2D] = None
    general_segments: PrimitiveList = None  # before sparsification


@dataclass
class AerialSegmentationResult:
    submaps: Dict[Crop, Submap]
    intermediates: Optional[Dict[Crop, AerialPatchIntermediates]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sub_patch_crops(patch_crop, seg_size_px):
    """Tile a patch crop into overlapping sub-patch crops.

    Returns [(x1, y1, x2, y2), ...] in full-image pixel coords.
    If seg_size_px >= patch dimension, returns [patch_crop].
    """
    x1, y1, x2, y2 = patch_crop
    pw, ph = x2 - x1, y2 - y1

    def _offsets(total, tile):
        if tile >= total:
            return [0]
        n = math.ceil(total / tile)
        stride = (total - tile) / (n - 1)
        return [round(stride * k) for k in range(n)]

    crops = []
    for dy in _offsets(ph, seg_size_px):
        for dx in _offsets(pw, seg_size_px):
            crops.append(
                (x1 + dx, y1 + dy, x1 + dx + seg_size_px, y1 + dy + seg_size_px)
            )
    return crops


# ---------------------------------------------------------------------------
# Per-patch conversion (parallelized across patches in Phase 2)
# ---------------------------------------------------------------------------

# Set once per patch-pool worker by the initializer so the converter is shipped
# a single time per worker instead of re-pickled on every submitted patch.
_WORKER_CONVERTER: Optional[SegmentToPrimitiveConverter] = None


def _patch_worker_init(converter):
    """ProcessPoolExecutor initializer for the per-patch conversion pool.

    Caps BLAS/OpenMP intra-op threads to 1 so N patch workers don't each spin up a
    full thread pool and oversubscribe the cores (the classify inside convert()
    runs serially in workers, so 1 process per patch is the whole story), and
    stashes the shared converter as a module global.
    """
    global _WORKER_CONVERTER
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    _WORKER_CONVERTER = converter


def _convert_patch(
    converter: SegmentToPrimitiveConverter,
    i_idx,
    j_idx,
    crop,
    sub_patch_results,
    pixel_len_m,
    border_dist_m,
    pose_flu,
    patch_size_m,
    parallel_classify,
):
    """Convert one patch's sub-crop segmentations into a sparse (descriptor-less)
    Submap. Pure numpy/shapely — no CUDA — so it is safe to run in a process pool.

    Returns ``(crop, submap, primitives)``; ``primitives`` is the
    pre-sparsification PrimitiveList kept for optional intermediates. Place
    recognition (GPU) is intentionally left to the caller / main process.
    """
    all_primitives = PrimitiveList()
    for aerial_segments, sub_crop in sub_patch_results:
        primitives = converter.convert(
            aerial_segments,
            pixel_len_m=pixel_len_m,
            crop=sub_crop,
            border_dist_m=border_dist_m,
            convert_to_infinite=False,
            parallel=parallel_classify,
        )
        all_primitives = all_primitives + primitives

    # Merge across all sub-patches, then convert long lines to infinite
    sparse_primitives = converter._cleanup_and_merge(
        all_primitives, convert_to_infinite=True
    )

    submap = Submap(
        id=(i_idx, j_idx),
        time=0.0,
        segments=sparse_primitives,
        pose=pose_flu,
        segment_frame=FrameType.IMG_PATCH_TOP_LEFT_CORNER,
        descriptor=None,
        metadata={
            "crop_center_m": np.array(
                [
                    (i_idx + 0.5) * patch_size_m,
                    -(j_idx + 0.5) * patch_size_m,
                ]
            )
        },
    )
    return crop, submap, all_primitives


def _convert_patch_worker(task):
    """Pool entry point: convert one patch using the worker-global converter, with
    the per-segment classify forced serial so we never nest process pools."""
    return _convert_patch(_WORKER_CONVERTER, *task, parallel_classify=False)


# ---------------------------------------------------------------------------
# AerialPatchPrimitiveMapping
# ---------------------------------------------------------------------------


class AerialPatchPrimitiveMapping:
    """Creates sparse primitive submaps from aerial image patches."""

    def __init__(
        self,
        patch_params: AerialPatchParams,
        converter: SegmentToPrimitiveConverter,
        aerial_segmenter: AerialSegmenter,
        place_recognition=None,
    ):
        self.patch_params = patch_params
        self.converter = converter
        self.aerial_segmenter = aerial_segmenter
        self.place_recognition = place_recognition

    def run(
        self,
        img: np.ndarray,
        img_origin: np.ndarray = None,
        return_intermediates: bool = False,
        show_progress: bool = False,
    ) -> AerialSegmentationResult:
        """Segment an aerial image into patch-based sparse primitive submaps.

        Args:
            img: Full aerial image (BGR).
            img_origin: UTM origin of the image top-left corner.
            return_intermediates: Whether to return per-patch intermediates.
            show_progress: Whether to show progress bars.

        Returns:
            AerialSegmentationResult with submaps and optional intermediates.
        """
        h, w = img.shape[:2]
        pixel_len_m = self.aerial_segmenter.params.pixel_len_m
        px_per_m = 1.0 / pixel_len_m

        patch_size_m = self.patch_params.aerial_img_patch_side_len_m
        patch_size_px = int(patch_size_m * px_per_m)

        seg_size_m = self.patch_params.aerial_img_segmentation_side_len_m
        if seg_size_m is None:
            seg_size_m = patch_size_m
        seg_size_px = int(seg_size_m * px_per_m)

        overlap = self.patch_params.aerial_img_patch_overlap
        stride = int(patch_size_px * (1.0 - overlap))

        if stride <= 0:
            raise ValueError("Patch overlap too large; stride becomes non-positive.")

        logger.info(
            f"Aerial batch: img={w}x{h}px, pixel_len_m={pixel_len_m:.6f}, "
            f"patch={patch_size_px}px ({patch_size_m}m), stride={stride}px, overlap={overlap}"
        )

        # Build FLU pose
        pose_flu = np.eye(4)
        pose_flu[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
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
        for j_idx, y1, i_idx, x1 in seg_iterator:
            crop = (x1, y1, x1 + patch_size_px, y1 + patch_size_px)
            patch_img = img[y1 : y1 + patch_size_px, x1 : x1 + patch_size_px].copy()

            # Reject thin/elongated segments
            thresh = self.converter.params.aerial_segment_line_rejection_thresh_m
            sub_crops = _compute_sub_patch_crops(crop, seg_size_px)
            sub_patch_results = []
            for sub_crop in sub_crops:
                aerial_segments = self.aerial_segmenter.run(img, crop=sub_crop)
                if thresh is not None:
                    aerial_segments = [
                        seg
                        for seg in aerial_segments
                        if not (
                            seg.obb_extents[0] < thresh[0]
                            and seg.obb_extents[1] > thresh[1]
                        )
                    ]
                sub_patch_results.append((aerial_segments, sub_crop))
            segmentation_results.append(
                (j_idx, y1, i_idx, x1, crop, patch_img, sub_patch_results)
            )

        # Phase 2: sparse conversion. Each patch is an independent submap, so
        # parallelize *across patches* (the CPU-bound line-merge dominates here),
        # forcing the per-segment classify inside convert() to run serially so we
        # use sparse_conversion_max_threads patch processes, not that many squared.
        # patch_img is only needed for intermediates/viz, not conversion, so keep
        # it main-side (keyed by crop) instead of shipping it through the pool.
        border_dist_m = self.patch_params.aerial_min_dist_to_border_m
        patch_img_by_crop = {r[4]: r[5] for r in segmentation_results}
        sub_patch_by_crop = {r[4]: r[6] for r in segmentation_results}

        def _task(r):
            j_idx, y1, i_idx, x1, crop, patch_img, sub_patch_results = r
            return (
                i_idx,
                j_idx,
                crop,
                sub_patch_results,
                pixel_len_m,
                border_dist_m,
                pose_flu,
                patch_size_m,
            )

        max_workers = self.converter.params.sparse_conversion_max_threads
        patch_results = []  # list of (crop, submap, primitives)
        if max_workers > 1 and len(segmentation_results) > 1:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_patch_worker_init,
                initargs=(self.converter,),
            ) as executor:
                futures = [
                    executor.submit(_convert_patch_worker, _task(r))
                    for r in segmentation_results
                ]
                fut_iter = as_completed(futures)
                if show_progress:
                    fut_iter = tqdm(
                        fut_iter, total=len(futures), desc="Aerial post-processing"
                    )
                for fut in fut_iter:
                    patch_results.append(fut.result())
        else:
            # One patch (or single worker): run serially but keep classify parallel
            # so a lone large patch still uses all the cores.
            seg_iterator = segmentation_results
            if show_progress:
                seg_iterator = tqdm(seg_iterator, desc="Aerial post-processing")
            for r in seg_iterator:
                patch_results.append(
                    _convert_patch(self.converter, *_task(r), parallel_classify=True)
                )

        # Place recognition (GPU) stays serial in the main process; then assemble
        # submaps and (optionally) intermediates, reattaching main-side patch_img.
        for crop, submap, primitives in patch_results:
            if self.place_recognition is not None:
                submap.descriptor = self.place_recognition.aerial_descriptor(
                    submap,
                    aerial_segmenter=self.aerial_segmenter,
                    img_bgr=img,
                    crop=crop,
                )
            submaps[crop] = submap
            if return_intermediates:
                all_aerial_segments = [
                    seg for segs, _ in sub_patch_by_crop[crop] for seg in segs
                ]
                intermediates[crop] = AerialPatchIntermediates(
                    patch_img=patch_img_by_crop[crop],
                    aerial_segments=all_aerial_segments,
                    general_segments=primitives,
                )

        return AerialSegmentationResult(submaps=submaps, intermediates=intermediates)
