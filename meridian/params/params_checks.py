"""Checks that span more than one params class, or params vs. command-line args.

Single-field normalization and validation (lowercasing, expandvars, allowed
values, legacy names) stays in each params class's __post_init__; this module is
for consistency between classes, so the params files stay mostly lists of
names and values.
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def require_frame_descriptor(segmenter_params):
    """Raise if a segmenter feeding cross-view place recognition has no
    frame_descriptor. Call at setup in pipelines that compute descriptors.

    Args:
        segmenter_params: SegmenterParams or AerialSegmenterParams
    """
    if segmenter_params.frame_descriptor is None:
        raise ValueError(
            f"{segmenter_params.params_key}.frame_descriptor is unset, but "
            "cross-view place recognition compares frame descriptors."
        )


def resolve_aerial_dir(aerial_dir, aerial_primitives_dir, required=False):
    """Reconcile the aerial submap dir from a --aerial flag vs. aerial_primitives_dir.

    Both point at the parent dir containing segments/. If both are set, --aerial
    wins (with a loud warning); returns whichever is provided (or None). With
    required=True, errors if neither is set.
    """
    if aerial_dir is not None and aerial_primitives_dir is not None:
        yellow, reset = "\033[1;33m", "\033[0m"
        bar = "=" * 80
        print(
            f"{yellow}{bar}\n"
            "WARNING: aerial submap directory set by both --aerial and the "
            "`aerial_primitives_dir` param.\n"
            f"  Loading from --aerial:                  {aerial_dir}\n"
            f"  Ignoring `aerial_primitives_dir` param: {aerial_primitives_dir}\n"
            f"{bar}{reset}",
            file=sys.stderr,
            flush=True,
        )
    resolved = aerial_dir if aerial_dir is not None else aerial_primitives_dir
    if required and resolved is None:
        raise ValueError(
            "No aerial submap directory. Pass --aerial <dir> or set "
            "`aerial_primitives_dir` in the cross_view_localization_data params "
            "(a directory containing segments/*.pkl)."
        )
    return resolved


def resolve_T_camera_flu(T_camera_flu, camera_pose_data, max_rot_diff_deg=1.0):
    """Resolve the camera -> body (FLU) transform: T_odom_body = T_odom_camera @ T_camera_flu.

    camera_pose_data.T_postmultiply maps body odometry to camera poses (it is
    T_body_camera), so if T_camera_flu is unset it defaults to
    inv(T_postmultiply), the robot's body / base_link frame. If both are set
    and their rotations differ by more than max_rot_diff_deg, warns loudly
    (the explicit T_camera_flu is kept).

    Args:
        T_camera_flu: 4x4 array or None (cross_view_localization_data.T_camera_flu)
        camera_pose_data: segment_mapping_data.camera_pose_data dict, or None
        max_rot_diff_deg: rotation disagreement above which to warn

    Returns:
        4x4 T_camera_flu, or None if neither source is set.
    """
    T_post = (camera_pose_data or {}).get("T_postmultiply")
    if T_post is None:
        return T_camera_flu
    T_camera_body = np.linalg.inv(np.asarray(T_post, dtype=float).reshape((4, 4)))
    if T_camera_flu is None:
        return T_camera_body
    R_diff = T_camera_flu[:3, :3].T @ T_camera_body[:3, :3]
    angle = np.degrees(Rot.from_matrix(R_diff).magnitude())
    if angle > max_rot_diff_deg:
        yellow, reset = "\033[1;33m", "\033[0m"
        bar = "=" * 80
        print(
            f"{yellow}{bar}\n"
            f"WARNING: T_camera_flu disagrees with camera_pose_data.T_postmultiply "
            f"by {angle:.1f} deg.\n"
            "  Using `cross_view_localization_data.T_camera_flu` as given. Remove it "
            "to use the body frame\n"
            "  from inv(`segment_mapping_data.camera_pose_data.T_postmultiply`).\n"
            f"  T_camera_flu:\n{np.array2string(T_camera_flu, precision=4)}\n"
            f"  inv(T_postmultiply):\n{np.array2string(T_camera_body, precision=4)}\n"
            f"{bar}{reset}",
            file=sys.stderr,
            flush=True,
        )
    return T_camera_flu
