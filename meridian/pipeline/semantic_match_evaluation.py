"""
Pipeline for evaluating how discriminative segment semantic features are.

For every sampled true geometric match (ground↔aerial point or line), records:
  (1) cosine similarity of the true pair
  (2) cosine sim of the ground segment against every other aerial segment (all individual scores)
  (3) cosine sim of the aerial segment against every other ground segment (all individual scores)

Histograms (PDF-normalised), statistics, and raw arrays are saved to the output directory.
"""

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from robotdatapy.data import PoseData

from meridian.map3d.submap import FrameType, Submap
from meridian.params.data_params import SemanticMatchEvaluationDataParams
from meridian.params.pipeline_params import SemanticMatchEvaluationParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate frame helpers
# ---------------------------------------------------------------------------


def _ground_seg_to_utm_2d(seg, submap: Submap, gt_pose_data: PoseData):
    """Return a 2-D (XY) UTM copy of a ground segment."""
    if submap.segment_frame == FrameType.CAMERA:
        # segments are in camera frame at observation time — use segment mean time
        t_mean = (seg.first_seen + seg.last_seen) / 2.0
        T = gt_pose_data.pose(t_mean)
    elif submap.segment_frame == FrameType.ODOMETRY:
        # segments are already in odom frame; derive T_utm_odom from submap time +
        # stored T_odom_camera in metadata
        T_utm_cam = gt_pose_data.pose(submap.time)
        T_odom_cam = submap.camera_pose
        T = T_utm_cam @ np.linalg.inv(T_odom_cam)
    else:
        raise ValueError(
            f"Unsupported ground segment_frame: {submap.segment_frame}. "
            "Expected CAMERA or ODOMETRY."
        )
    seg_utm = seg.to_dim(3).copy()
    seg_utm.transform(T)
    return seg_utm.to_dim(2)


def _aerial_seg_to_utm_2d(seg, submap: Submap):
    """Return a 2-D (XY) UTM copy of an aerial segment."""
    seg_utm = seg.to_dim(3).copy()
    seg_utm.transform(submap.pose)
    return seg_utm.to_dim(2)


def _pad_to_homogeneous(pt):
    """Pad a 2D or 3D point to a 4-element homogeneous vector."""
    pt = np.asarray(pt).flatten()
    if len(pt) == 2:
        return np.array([pt[0], pt[1], 0.0, 1.0])
    return np.array([pt[0], pt[1], pt[2], 1.0])


def _ground_submap_utm_centroid_xy(submap: Submap, gt_pose_data: PoseData):
    """Return the 2-D XY UTM centroid of a ground submap's segments."""
    if submap.segment_frame == FrameType.CAMERA:
        T = gt_pose_data.pose(submap.time)
    elif submap.segment_frame == FrameType.ODOMETRY:
        T_utm_cam = gt_pose_data.pose(submap.time)
        T_odom_cam = submap.camera_pose
        T = T_utm_cam @ np.linalg.inv(T_odom_cam)
    else:
        raise ValueError(f"Unsupported ground segment_frame: {submap.segment_frame}")
    pts = np.array([seg.point.flatten() for seg in submap.segments])
    centroid_local = pts.mean(axis=0)
    centroid_utm = (T @ _pad_to_homogeneous(centroid_local))[:2]
    return centroid_utm


def _aerial_submap_utm_centroid_xy(submap: Submap):
    """Return the 2-D XY UTM centroid of an aerial submap's segments."""
    pts = np.array([seg.point.flatten() for seg in submap.segments])
    centroid_local = pts.mean(axis=0)
    centroid_utm = (submap.pose @ _pad_to_homogeneous(centroid_local))[:2]
    return centroid_utm


def _seg_point_2d(seg_utm):
    """Return the 2-D XY position of a segment's representative point."""
    return seg_utm.point.flatten()[:2]


# ---------------------------------------------------------------------------
# Line geometry helpers
# ---------------------------------------------------------------------------


def _line_min_dist(line1, line2) -> float:
    """Minimum distance between two LinePrimitive objects (3-D)."""
    return line1.min_dist_to(line2)


def _line_angle_diff_deg(line1, line2) -> float:
    """Angular difference in degrees between two line directions (0–90°)."""
    dot = abs(np.dot(line1.direction.flatten(), line2.direction.flatten()))
    dot = np.clip(dot, 0.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


# ---------------------------------------------------------------------------
# Flat feature arrays for all segments
# ---------------------------------------------------------------------------


def _build_feature_array(submaps, get_fn):
    """
    Build a flat (N, D) array of cos_features and a matching index list of
    (submap_key, seg_id) tuples, for all segments returned by `get_fn(submap)`.
    Segments without a cos_feature are skipped.
    """
    feats = []
    index = []  # list of (submap_key, seg_id)
    for key, sm in submaps.items():
        for seg in get_fn(sm):
            if seg.cos_feature is None:
                continue
            feats.append(seg.cos_feature.flatten())
            index.append((key, seg.id))
    if not feats:
        return np.empty((0, 0)), index
    return np.vstack(feats), index


# ---------------------------------------------------------------------------
# Match recording
# ---------------------------------------------------------------------------


def _compute_sims(
    g_feat, a_feat, all_ground_feats, all_aerial_feats, g_global_idx, a_global_idx
):
    """Return (sim_match, sims_nonmatch) where non-match sims are combined."""
    sim_match = float(np.dot(g_feat, a_feat))

    # ground segment vs all OTHER aerial segments
    mask_a = np.ones(len(all_aerial_feats), dtype=bool)
    mask_a[a_global_idx] = False
    sims_g_nonmatch = all_aerial_feats[mask_a] @ g_feat  # shape (N_aerial-1,)

    # aerial segment vs all OTHER ground segments
    mask_g = np.ones(len(all_ground_feats), dtype=bool)
    mask_g[g_global_idx] = False
    sims_a_nonmatch = all_ground_feats[mask_g] @ a_feat  # shape (N_ground-1,)

    return sim_match, np.concatenate([sims_g_nonmatch, sims_a_nonmatch])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def semantic_match_evaluation(params_path: str, output_dir: str, run: str = None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- load params --------------------------------------------------------
    data_params = SemanticMatchEvaluationDataParams.load(params_path, run=run)
    params = SemanticMatchEvaluationParams.load(params_path, run=run)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load submaps -------------------------------------------------------
    logger.info("Loading ground submaps from %s", data_params.ground_submap_dir)
    ground_submaps = {
        p.stem: Submap.load(p)
        for p in sorted(Path(data_params.ground_submap_dir).glob("*.pkl"))
    }
    logger.info("Loading aerial submaps from %s", data_params.aerial_submap_dir)
    aerial_submaps = {
        p.stem: Submap.load(p)
        for p in sorted(Path(data_params.aerial_submap_dir).glob("*.pkl"))
    }
    logger.info(
        "Loaded %d ground, %d aerial submaps",
        len(ground_submaps),
        len(aerial_submaps),
    )

    # --- load GT trajectory -------------------------------------------------
    gt_pose_data = PoseData.from_dict(data_params.gt_pose_data)

    # --- build flat feature arrays ------------------------------------------
    logger.info("Building flat feature arrays for non-match similarity computation...")
    all_ground_point_feats, ground_point_index = _build_feature_array(
        ground_submaps, lambda sm: sm.segments.get_points()
    )
    all_aerial_point_feats, aerial_point_index = _build_feature_array(
        aerial_submaps, lambda sm: sm.segments.get_points()
    )
    all_ground_line_feats, ground_line_index = _build_feature_array(
        ground_submaps, lambda sm: sm.segments.get_lines()
    )
    all_aerial_line_feats, aerial_line_index = _build_feature_array(
        aerial_submaps, lambda sm: sm.segments.get_lines()
    )

    # fast lookup: (submap_key, seg_id) -> global index
    def _build_lookup(index):
        return {item: i for i, item in enumerate(index)}

    g_pt_lookup = _build_lookup(ground_point_index)
    a_pt_lookup = _build_lookup(aerial_point_index)
    g_ln_lookup = _build_lookup(ground_line_index)
    a_ln_lookup = _build_lookup(aerial_line_index)

    # --- sample true matches ------------------------------------------------
    point_match_sims = []
    point_nonmatch_sims = []

    line_match_sims = []
    line_nonmatch_sims = []

    n_point_target = params.num_point_matches
    n_line_target = params.num_line_matches

    # randomise submap-pair iteration order
    submap_pairs = [(gk, ak) for gk in ground_submaps for ak in aerial_submaps]
    random.shuffle(submap_pairs)

    with tqdm(total=n_point_target + n_line_target, desc="Finding matches") as pbar:
        for g_key, a_key in submap_pairs:
            if (
                len(point_match_sims) >= n_point_target
                and len(line_match_sims) >= n_line_target
            ):
                break

            g_sm = ground_submaps[g_key]
            a_sm = aerial_submaps[a_key]

            # ---- skip non-overlapping submap pairs -------------------------
            try:
                g_xy = _ground_submap_utm_centroid_xy(g_sm, gt_pose_data)
                a_xy = _aerial_submap_utm_centroid_xy(a_sm)
                submap_dist = np.linalg.norm(g_xy - a_xy)
                if submap_dist > params.submap_max_dist_m:
                    continue
            except Exception:
                continue

            # ---- POINTS ----------------------------------------------------
            if len(point_match_sims) < n_point_target:
                g_points = list(g_sm.segments.get_points())
                a_points = list(a_sm.segments.get_points())
                random.shuffle(g_points)

                for g_seg in g_points:
                    if len(point_match_sims) >= n_point_target:
                        break
                    if g_seg.cos_feature is None:
                        continue
                    # transform ground segment to UTM
                    try:
                        g_utm = _ground_seg_to_utm_2d(g_seg, g_sm, gt_pose_data)
                    except Exception:
                        continue
                    g_xy = _seg_point_2d(g_utm)

                    # pre-transform all aerial points for this submap
                    a_utms = [_aerial_seg_to_utm_2d(a, a_sm) for a in a_points]
                    a_xys = [_seg_point_2d(a) for a in a_utms]

                    random.shuffle(paired := list(zip(a_points, a_utms, a_xys)))
                    for a_seg, a_seg_utm, a_xy in paired:
                        if a_seg.cos_feature is None:
                            continue
                        dist = np.linalg.norm(g_xy - a_xy)
                        if dist > params.point_match_max_dist_m:
                            continue

                        # disqualify if another aerial point is too close to g_seg
                        if any(
                            np.linalg.norm(g_xy - o_xy) < params.point_other_min_dist_m
                            for o, _, o_xy in paired
                            if o.id != a_seg.id
                        ):
                            continue

                        # disqualify if another ground point is too close to a_seg
                        other_ground_close = False
                        for o_g in g_points:
                            if o_g.id == g_seg.id or o_g.cos_feature is None:
                                continue
                            try:
                                o_g_utm = _ground_seg_to_utm_2d(o_g, g_sm, gt_pose_data)
                            except Exception:
                                continue
                            if (
                                np.linalg.norm(_seg_point_2d(o_g_utm) - a_xy)
                                < params.point_other_min_dist_m
                            ):
                                other_ground_close = True
                                break
                        if other_ground_close:
                            continue

                        # valid match — compute and record
                        g_idx = g_pt_lookup.get((g_key, g_seg.id))
                        a_idx = a_pt_lookup.get((a_key, a_seg.id))
                        if g_idx is None or a_idx is None:
                            continue
                        sm, s_nm = _compute_sims(
                            g_seg.cos_feature,
                            a_seg.cos_feature,
                            all_ground_point_feats,
                            all_aerial_point_feats,
                            g_idx,
                            a_idx,
                        )
                        point_match_sims.append(sm)
                        point_nonmatch_sims.append(s_nm)
                        pbar.update(1)
                        break  # one match per g_seg

            # ---- LINES -----------------------------------------------------
            if len(line_match_sims) < n_line_target:
                g_lines = list(g_sm.segments.get_lines())
                a_lines = list(a_sm.segments.get_lines())

                random.shuffle(g_lines)

                for g_seg in g_lines:
                    if len(line_match_sims) >= n_line_target:
                        break
                    if g_seg.cos_feature is None:
                        continue
                    try:
                        g_utm = _ground_seg_to_utm_2d(g_seg, g_sm, gt_pose_data)
                    except Exception:
                        continue

                    a_utms = [_aerial_seg_to_utm_2d(a, a_sm) for a in a_lines]

                    paired_lines = list(zip(a_lines, a_utms))
                    random.shuffle(paired_lines)

                    for a_seg, a_seg_utm in paired_lines:
                        if a_seg.cos_feature is None:
                            continue

                        dist = _line_min_dist(g_utm, a_seg_utm)
                        ang = _line_angle_diff_deg(g_utm, a_seg_utm)

                        if dist > params.line_match_max_dist_m:
                            continue
                        if ang > params.line_match_max_ang_diff_deg:
                            continue

                        # disqualify if another aerial line also matches g_seg
                        ambiguous = False
                        for o_a, o_a_utm in paired_lines:
                            if o_a.id == a_seg.id:
                                continue
                            o_dist = _line_min_dist(g_utm, o_a_utm)
                            o_ang = _line_angle_diff_deg(g_utm, o_a_utm)
                            if (
                                o_dist <= params.line_other_min_dist_m
                                and o_ang <= params.line_other_min_ang_diff_deg
                            ):
                                ambiguous = True
                                break
                        if ambiguous:
                            continue

                        # disqualify if another ground line also matches a_seg
                        for o_g in g_lines:
                            if o_g.id == g_seg.id or o_g.cos_feature is None:
                                continue
                            try:
                                o_g_utm = _ground_seg_to_utm_2d(o_g, g_sm, gt_pose_data)
                            except Exception:
                                continue
                            o_dist = _line_min_dist(o_g_utm, a_seg_utm)
                            o_ang = _line_angle_diff_deg(o_g_utm, a_seg_utm)
                            if (
                                o_dist <= params.line_other_min_dist_m
                                and o_ang <= params.line_other_min_ang_diff_deg
                            ):
                                ambiguous = True
                                break
                        if ambiguous:
                            continue

                        g_idx = g_ln_lookup.get((g_key, g_seg.id))
                        a_idx = a_ln_lookup.get((a_key, a_seg.id))
                        if g_idx is None or a_idx is None:
                            continue
                        sm, s_nm = _compute_sims(
                            g_seg.cos_feature,
                            a_seg.cos_feature,
                            all_ground_line_feats,
                            all_aerial_line_feats,
                            g_idx,
                            a_idx,
                        )
                        line_match_sims.append(sm)
                        line_nonmatch_sims.append(s_nm)
                        pbar.update(1)
                        break

    logger.info(
        "Collected %d point matches and %d line matches",
        len(point_match_sims),
        len(line_match_sims),
    )

    # flatten per-match non-match arrays into a single array each
    point_nonmatch_flat = (
        np.concatenate(point_nonmatch_sims) if point_nonmatch_sims else np.array([])
    )
    line_nonmatch_flat = (
        np.concatenate(line_nonmatch_sims) if line_nonmatch_sims else np.array([])
    )

    # --- save outputs -------------------------------------------------------
    _save_histogram(
        output_dir / "point_similarities.png",
        point_match_sims,
        point_nonmatch_flat,
        title="Point semantic similarity",
    )
    _save_histogram(
        output_dir / "line_similarities.png",
        line_match_sims,
        line_nonmatch_flat,
        title="Line semantic similarity",
    )

    _save_statistics(
        output_dir / "statistics.txt",
        point_match_sims,
        point_nonmatch_flat,
        line_match_sims,
        line_nonmatch_flat,
    )

    np.savez(
        output_dir / "point_similarities.npz",
        match=np.array(point_match_sims),
        nonmatch=point_nonmatch_flat,
    )
    np.savez(
        output_dir / "line_similarities.npz",
        match=np.array(line_match_sims),
        nonmatch=line_nonmatch_flat,
    )
    logger.info("Outputs written to %s", output_dir)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_histogram(path, match_sims, nonmatch_sims, title=""):
    fig, ax = plt.subplots()
    bins = 30
    if len(match_sims):
        ax.hist(
            match_sims, bins=bins, alpha=0.6, density=True, label="match", color="green"
        )
    if len(nonmatch_sims):
        ax.hist(
            nonmatch_sims,
            bins=bins,
            alpha=0.6,
            density=True,
            label="non-match",
            color="orange",
        )
    ax.legend()
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _dist_stats(data):
    a = np.asarray(data)
    if a.size == 0:
        return "  (no data)"
    return (
        f"  mean={a.mean():.4f}  median={np.median(a):.4f}"
        f"  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}"
    )


def _save_statistics(
    path,
    pt_match,
    pt_nm,
    ln_match,
    ln_nm,
):
    lines = [
        "=== Points ===",
        f"match:      {_dist_stats(pt_match)}",
        f"non-match:  {_dist_stats(pt_nm)}",
        "",
        "=== Lines ===",
        f"match:      {_dist_stats(ln_match)}",
        f"non-match:  {_dist_stats(ln_nm)}",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic match evaluation pipeline")
    parser.add_argument(
        "-p", "--params", required=True, help="Path to params YAML file or directory"
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--run", default=None, help="Run key within params file")
    args = parser.parse_args()
    semantic_match_evaluation(args.params, args.output, args.run)
