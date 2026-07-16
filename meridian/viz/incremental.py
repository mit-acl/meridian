"""Standalone plotting helpers for the incremental cross-view pipeline.

These render end-of-run diagnostics (estimated-vs-GT trajectory on the aerial
image, localization error vs. time) from a list of ``(time, T_utm_body)``
instantaneous poses. They are pure rendering: the pipeline owns the pose history
and calls these to produce figures. The incremental *movie* lives separately in
``incremental_movie.py``.
"""

import logging
import pathlib
from typing import Callable, List, Tuple, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

InstantPoseHistory = List[Tuple[float, np.ndarray]]


def make_utm_to_pixel(data) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function mapping UTM XY -> aerial image (col, row).

    Uses the GeoTIFF transform when available, otherwise a simple origin +
    scale conversion. Shared by the trajectory plot and the movie writer.
    """
    if data.geotiff_transform is not None:

        def utm_to_pixel(xy: np.ndarray) -> np.ndarray:
            return data.aerial_utm_to_pixel(np.atleast_2d(xy))
    else:
        origin_x, origin_y = data.aerial_img_origin
        pixel_len_m = data.aerial_img_scale

        def utm_to_pixel(xy: np.ndarray) -> np.ndarray:
            xy = np.atleast_2d(xy)
            cols = (xy[:, 0] - origin_x) / pixel_len_m
            rows = (origin_y - xy[:, 1]) / pixel_len_m
            return np.column_stack([cols, rows])

    return utm_to_pixel


def plot_incremental_trajectory(
    instant_pose_history: InstantPoseHistory,
    data,
    viz_params,
    out_path: Union[str, pathlib.Path],
):
    """Draw the estimated incremental trajectory (and GT, if available) over the
    aerial image. Segments split by NaN poses (pre-localization gaps) are drawn
    as separate polylines."""
    if not instant_pose_history:
        return

    positions = []
    for _, T in instant_pose_history:
        if np.any(np.isnan(T)):
            positions.append(None)
        else:
            positions.append(T[:2, 3])

    # Split into contiguous non-NaN runs.
    segments_xy = []
    cur = []
    for p in positions:
        if p is None:
            if cur:
                segments_xy.append(np.array(cur))
                cur = []
        else:
            cur.append(p)
    if cur:
        segments_xy.append(np.array(cur))

    utm_to_pixel = make_utm_to_pixel(data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    aerial_img = data.aerial_img
    ds = max(1, min(aerial_img.shape[0], aerial_img.shape[1]) // 2000)
    ax.imshow(
        cv.cvtColor(aerial_img[::ds, ::ds], cv.COLOR_BGR2RGB),
        extent=[0, aerial_img.shape[1], aerial_img.shape[0], 0],
    )

    for i, seg_xy in enumerate(segments_xy):
        if seg_xy.size == 0:
            continue
        seg_px = utm_to_pixel(seg_xy)
        label = "Estimated (incremental)" if i == 0 else None
        ax.plot(
            seg_px[:, 0],
            seg_px[:, 1],
            color=viz_params.estimated_trajectory_color,
            linestyle="-",
            linewidth=1.2,
            label=label,
        )

    if data.gt_pose_data is not None:
        gt_xy = []
        for t, _ in instant_pose_history:
            try:
                gt_pose = data.gt_pose_data.pose(t)
                if data.T_camera_flu is not None:
                    gt_body = gt_pose @ data.T_camera_flu
                else:
                    gt_body = gt_pose
                gt_xy.append(gt_body[:2, 3])
            except Exception:
                gt_xy.append([np.nan, np.nan])
        gt_xy = np.array(gt_xy)
        valid = ~np.any(np.isnan(gt_xy), axis=1)
        if np.any(valid):
            gt_px = utm_to_pixel(gt_xy[valid])
            ax.plot(
                gt_px[:, 0],
                gt_px[:, 1],
                color=viz_params.gt_trajectory_color,
                linestyle="-",
                linewidth=1.2,
                label="Ground Truth",
            )

    ax.legend()
    ax.set_title("Incremental Cross-View Localization")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_time(
    instant_pose_history: InstantPoseHistory,
    data,
    out_path: Union[str, pathlib.Path],
) -> dict:
    """Plot translation/heading error vs. time against GT and return summary
    metrics. Returns an empty dict when there is no GT or no pose history; a
    dict with ``n_valid``/``n_total`` (+ error stats when ``n_valid`` > 0)
    otherwise.

    Note: this both renders the plot and computes the error metrics consumed by
    the pipeline's results file; the metric computation is arguably an eval
    concern and could later be split into a pure helper.
    """
    if not instant_pose_history or data.gt_pose_data is None:
        return {}

    t0 = instant_pose_history[0][0]
    ts, trans_errs, yaw_errs = [], [], []
    for t, T in instant_pose_history:
        ts.append(t - t0)
        if np.any(np.isnan(T)):
            trans_errs.append(np.nan)
            yaw_errs.append(np.nan)
            continue
        try:
            gt_pose = data.gt_pose_data.pose(t)
        except Exception:
            trans_errs.append(np.nan)
            yaw_errs.append(np.nan)
            continue
        if data.T_camera_flu is not None:
            gt_body = gt_pose @ data.T_camera_flu
        else:
            gt_body = gt_pose
        trans_errs.append(float(np.linalg.norm(T[:2, 3] - gt_body[:2, 3])))
        est_yaw = np.arctan2(T[1, 0], T[0, 0])
        gt_yaw = np.arctan2(gt_body[1, 0], gt_body[0, 0])
        d = est_yaw - gt_yaw
        yaw_errs.append(float(np.abs(np.arctan2(np.sin(d), np.cos(d)))))

    ts = np.array(ts)
    trans_errs = np.array(trans_errs)
    yaw_errs = np.array(yaw_errs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ts, trans_errs, "-", color="tab:blue")
    axes[0].set_ylabel("Translation error (m)")
    axes[0].grid(True)
    axes[1].plot(ts, np.rad2deg(yaw_errs), "-", color="tab:orange")
    axes[1].set_ylabel("Heading error (deg)")
    axes[1].set_xlabel("t - t0 (s)")
    axes[1].grid(True)
    fig.suptitle("Incremental localization error vs. time")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    valid = ~np.isnan(trans_errs) & ~np.isnan(yaw_errs)
    n_valid = int(valid.sum())
    n_total = len(ts)
    if n_valid > 0:
        return {
            "n_valid": n_valid,
            "n_total": n_total,
            "rmse_trans_m": float(np.sqrt(np.mean(trans_errs[valid] ** 2))),
            "mean_trans_m": float(np.mean(trans_errs[valid])),
            "max_trans_m": float(np.max(trans_errs[valid])),
            "rmse_yaw_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_errs[valid] ** 2)))),
            "mean_yaw_deg": float(np.rad2deg(np.mean(yaw_errs[valid]))),
            "max_yaw_deg": float(np.rad2deg(np.max(yaw_errs[valid]))),
        }
    return {"n_valid": 0, "n_total": n_total}
