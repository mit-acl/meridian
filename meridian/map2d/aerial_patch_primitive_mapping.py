import logging
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
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
    # Wall-clock breakdown (s): total, segmentation (GPU loop), blocked
    # (segmentation paused on a full conversion pool), conversion_tail
    # (conversion still running after the last segmentation).
    timing: Optional[Dict[str, float]] = None


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
    ):
        self.patch_params = patch_params
        self.converter = converter
        self.aerial_segmenter = aerial_segmenter

    def run(
        self,
        img: np.ndarray,
        img_origin: np.ndarray = None,
        return_intermediates: bool = False,
        show_progress: bool = False,
        on_patch_done: Optional[
            Callable[[Crop, Submap, AerialPatchIntermediates], None]
        ] = None,
        max_pending_patches: Optional[int] = None,
    ) -> AerialSegmentationResult:
        """Segment an aerial image into patch-based sparse primitive submaps.

        Segmentation (GPU) runs serially in this process while the sparse
        conversion of already-segmented patches runs in a process pool in the
        background. At most `max_pending_patches` segmented-but-unconverted
        patches are held at once (their dense segments dominate memory, ~70 MB
        per 60 m patch at 0.055 m/px), so memory stays bounded regardless of
        image size; segmentation only pauses when the pool is saturated.

        Args:
            img: Full aerial image (BGR).
            img_origin: UTM origin of the image top-left corner.
            return_intermediates: Whether to return per-patch intermediates.
                These hold every patch's dense segments, so memory grows with
                the image (tens of GB for a large image); prefer on_patch_done.
            show_progress: Whether to show progress bars.
            on_patch_done: Optional callback, called in this process as each
                patch finishes, with (crop, submap, intermediates). The
                intermediates are released afterwards unless
                return_intermediates is set. Use it to save / visualize patches
                as they complete.
            max_pending_patches: Cap on patches in flight in the conversion pool
                (default 4 x sparse_conversion_max_threads).

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

        border_dist_m = self.patch_params.aerial_min_dist_to_border_m
        thresh = self.converter.params.aerial_segment_line_rejection_thresh_m
        keep_intermediates = return_intermediates or on_patch_done is not None

        def _segment_patch(crop):
            """GPU segmentation of one patch's sub-crops; drops thin/elongated
            segments."""
            sub_patch_results = []
            for sub_crop in _compute_sub_patch_crops(crop, seg_size_px):
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
            return sub_patch_results

        def _finish_patch(crop, submap, primitives, sub_patch_results):
            """Main-process tail of one patch: place-recognition descriptor
            (GPU), then hand off / keep the submap and intermediates."""
            submap.descriptor = self.aerial_segmenter.get_crop_descriptor(
                img, crop=crop
            )
            if submap.descriptor is None:
                raise RuntimeError(
                    f"Aerial patch {crop} got no place-recognition descriptor "
                    "(frame_descriptor="
                    f"{self.aerial_segmenter.frame_descriptor_type!r}). "
                    "aerial_segmenter.get_crop_descriptor returned None; check the "
                    "aerial_segmenter frame_descriptor and semantics settings."
                )
            CrossViewPlaceRecognition.tag(
                submap, self.aerial_segmenter.frame_descriptor_type
            )
            submaps[crop] = submap
            if not keep_intermediates:
                return
            x1, y1, x2, y2 = crop
            patch_intermediates = AerialPatchIntermediates(
                patch_img=img[y1:y2, x1:x2].copy(),
                aerial_segments=[seg for segs, _ in sub_patch_results for seg in segs],
                general_segments=primitives,
            )
            if on_patch_done is not None:
                on_patch_done(crop, submap, patch_intermediates)
            if return_intermediates:
                intermediates[crop] = patch_intermediates

        def _task(i_idx, j_idx, crop, sub_patch_results):
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
        if max_pending_patches is None:
            max_pending_patches = 4 * max(1, max_workers)

        progress = tqdm(
            total=len(patches), desc="Aerial patches", disable=not show_progress
        )
        t_start = time.time()
        t_segment = 0.0  # GPU segmentation
        t_blocked = 0.0  # segmentation paused because the pool was saturated

        if max_workers > 1 and len(patches) > 1:
            # Segment serially here; convert in the background across patches,
            # forcing the per-segment classify inside convert() to run serially
            # so we use max_workers processes, not that many squared. Only the
            # segments ship to the pool; patch_img stays main-side.
            pending = {}  # future -> sub_patch_results (for intermediates)

            def _collect(done):
                for fut in done:
                    sub_patch_results = pending.pop(fut)
                    crop, submap, primitives = fut.result()
                    _finish_patch(crop, submap, primitives, sub_patch_results)
                    progress.update(1)

            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_patch_worker_init,
                initargs=(self.converter,),
            ) as executor:
                for j_idx, y1, i_idx, x1 in patches:
                    crop = (x1, y1, x1 + patch_size_px, y1 + patch_size_px)
                    t0 = time.time()
                    sub_patch_results = _segment_patch(crop)
                    t_segment += time.time() - t0
                    fut = executor.submit(
                        _convert_patch_worker,
                        _task(i_idx, j_idx, crop, sub_patch_results),
                    )
                    pending[fut] = sub_patch_results if keep_intermediates else None
                    del sub_patch_results

                    # Collect whatever has finished without waiting; block only
                    # when too many patches are in flight.
                    _collect([f for f in list(pending) if f.done()])
                    if len(pending) >= max_pending_patches:
                        t0 = time.time()
                        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                        t_blocked += time.time() - t0
                        _collect(done)
                t_seg_end = time.time()
                while pending:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    _collect(done)
        else:
            # One patch (or single worker): run serially but keep classify parallel
            # so a lone large patch still uses all the cores.
            for j_idx, y1, i_idx, x1 in patches:
                crop = (x1, y1, x1 + patch_size_px, y1 + patch_size_px)
                t0 = time.time()
                sub_patch_results = _segment_patch(crop)
                t_segment += time.time() - t0
                crop, submap, primitives = _convert_patch(
                    self.converter,
                    *_task(i_idx, j_idx, crop, sub_patch_results),
                    parallel_classify=True,
                )
                _finish_patch(crop, submap, primitives, sub_patch_results)
                progress.update(1)
            t_seg_end = time.time()
        progress.close()

        timing = {
            "total": time.time() - t_start,
            "segmentation": t_segment,
            "blocked": t_blocked,
            "conversion_tail": time.time() - t_seg_end,
        }
        logger.info(
            f"Aerial patches: {len(patches)} in {timing['total']:.1f} s; "
            f"segmentation {timing['segmentation']:.1f} s, paused on a full "
            f"conversion pool {timing['blocked']:.1f} s, conversion tail after "
            f"segmentation {timing['conversion_tail']:.1f} s"
        )

        return AerialSegmentationResult(
            submaps=submaps, intermediates=intermediates, timing=timing
        )
