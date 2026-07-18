"""Candidate construction for cross-view localization.

A "candidate" is one aerial<->ground loop-closure hypothesis, expressed as a set
of SE(2)/SE(3) transforms plus bookkeeping, ready to feed CLIPPER outlier
rejection and PGO (see `meridian.cross_view.rpgo`). This lives in the algorithm
layer (`cross_view`) so both the offline localization pipeline and the
incremental localizer can build candidates without depending on pipeline-layer
data types.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from meridian.cross_view.rpgo import se2_from_xytheta, se2_to_se3, se3_to_se2
from meridian.map3d.submap import Submap

if TYPE_CHECKING:
    from robotdatapy.data import PoseData
    from meridian.cross_view.matching import CrossViewMatchResult

logger = logging.getLogger(__name__)


@dataclass
class LocalizationContext:
    """Minimal coordinate/matching context the localization algorithm needs.

    Built by the pipeline from `CrossViewLocalizationData` (see
    `context_from_data` in `meridian.pipeline.cross_view_localization`); a ROS
    wrapper can construct it directly. Keeps the algorithm independent of any
    pipeline-layer data class.
    """

    T_camera_flu: Optional[np.ndarray]
    aerial_img_scale: float
    # Aerial-local (meters) -> image (col, row). Used by matching for aerial
    # patch selection. None when no GeoTIFF transform is available.
    local_to_pixel_fn: Optional[Callable] = None
    # Image (col, row) -> absolute UTM (x, y). Used by candidate construction
    # only when the aerial image needs CRS reprojection (native_crs != utm_crs);
    # None otherwise (candidates then use the raw transform translation).
    pixel_to_utm_fn: Optional[Callable] = None
    gt_pose_data: Optional["PoseData"] = None


def build_candidates_from_match_result(
    match_result: "CrossViewMatchResult",
    ground_submaps: Dict[str, Submap],
    aerial_submaps: Dict[str, Submap],
    context: LocalizationContext,
    min_assoc: int,
) -> List[dict]:
    """Build candidate list from an in-memory CrossViewMatchResult."""
    first_aerial_key = next(iter(aerial_submaps))
    pose_flu = aerial_submaps[first_aerial_key].pose

    candidates = []

    for ground_key, results_matrix in match_result.results.items():
        if ground_key not in ground_submaps:
            continue
        ground_submap = ground_submaps[ground_key]
        ground_camera_pose = ground_submap.camera_pose

        for idx in np.ndindex(results_matrix.shape):
            cell = results_matrix[idx]
            hypotheses = cell if isinstance(cell, list) else [cell]
            for result in hypotheses:
                if result.num_associations < min_assoc:
                    continue
                T_i_j_hat = result.T_i_j_hat
                if np.any(np.isnan(T_i_j_hat)):
                    continue

                if context.T_camera_flu is not None:
                    T_odom_ground = ground_camera_pose @ context.T_camera_flu
                else:
                    T_odom_ground = ground_camera_pose

                T_full = pose_flu @ T_i_j_hat

                if np.linalg.det(T_full[:2, :2]) < 0:
                    logger.debug(
                        "Rejecting candidate with reflected rotation "
                        f"(det(R_2x2)={np.linalg.det(T_full[:2, :2]):.4f})"
                    )
                    continue

                yaw = np.arctan2(T_full[1, 0], T_full[0, 0])

                pixel_len_m = context.aerial_img_scale
                body_col = T_i_j_hat[0, 3] / pixel_len_m
                body_row = T_i_j_hat[1, 3] / pixel_len_m
                if context.pixel_to_utm_fn is not None:
                    utm_x, utm_y = context.pixel_to_utm_fn(body_col, body_row)
                else:
                    utm_x = T_full[0, 3]
                    utm_y = T_full[1, 3]

                T_utm_body_se2 = se2_from_xytheta(utm_x, utm_y, yaw)

                T_utm_odom_4x4 = se2_to_se3(T_utm_body_se2) @ np.linalg.inv(
                    T_odom_ground
                )
                T_utm_odom_se2 = se3_to_se2(T_utm_odom_4x4)

                T_utm_odom_gt_se2 = None
                if context.gt_pose_data is not None:
                    try:
                        gt_pose = context.gt_pose_data.pose(ground_submap.time)
                        if context.T_camera_flu is not None:
                            gt_body = gt_pose @ context.T_camera_flu
                        else:
                            gt_body = gt_pose
                        T_utm_odom_gt = gt_body @ np.linalg.inv(T_odom_ground)
                        T_utm_odom_gt_se2 = se3_to_se2(T_utm_odom_gt)
                    except Exception:
                        pass

                candidates.append(
                    {
                        "T_utm_odom_se2": T_utm_odom_se2,
                        "T_utm_body_se2": T_utm_body_se2,
                        "T_i_j_hat": T_i_j_hat,
                        "T_i_j": result.T_i_j,
                        "aerial_pose": pose_flu,
                        "ground_camera_pose": ground_camera_pose,
                        "T_odom_ground": T_odom_ground,
                        "ground_key": ground_key,
                        "aerial_key": f"{idx[0]}_{idx[1]}",
                        "num_associations": result.num_associations,
                        "ground_submap_time": ground_submap.time,
                        "T_utm_odom_gt_se2": T_utm_odom_gt_se2,
                        "count": getattr(result, "count", 1),
                    }
                )

    return candidates
