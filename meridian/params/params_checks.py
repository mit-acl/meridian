"""Checks that span more than one params class, or params vs. command-line args.

Single-field normalization and validation (lowercasing, expandvars, allowed
values, legacy names) stays in each params class's __post_init__; this module is
for consistency between classes, so the params files stay mostly lists of
names and values. It also resolves the transforms of pose data params, which can be
looked up from a bag's TF instead of given as matrices (resolve_pose_data_transforms).
"""

import functools
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from robotdatapy.data import PoseData


# Transform lookups (pose data T_premultiply / T_postmultiply)
POSE_DATA_TRANSFORM_KEYS = ("T_premultiply", "T_postmultiply")
TRANSFORM_LOOKUP_KEYS = {"parent_frame", "child_frame", "path", "tf_topic_namespace"}
BAG_POSE_DATA_TYPES = ("bag", "bag_tf")
# dynamic TF links of a lookup are read over the start of the bag and must not move
TRANSFORM_LOOKUP_WINDOW_S = 10.0
RIGID_TRANS_TOL_M = 1e-3
RIGID_ROT_TOL_DEG = 0.1

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


# Transform lookups (pose data T_premultiply / T_postmultiply)

@functools.lru_cache(maxsize=None)
def lookup_bag_transform(path, parent_frame, child_frame, tf_topic_namespace=None):
    """T_parent_child from a bag's TF (/tf and /tf_static, or /<tf_topic_namespace>/tf and
    /<tf_topic_namespace>/tf_static). parent_frame must be an ancestor of child_frame. The
    chain may include dynamic TF links if they are rigid: they are read over the first 
    TRANSFORM_LOOKUP_WINDOW_S of the bag and must not move.
    """

    # only pass the namespace when set, so robotdatapy < 1.1.19 works for un-namespaced TF
    tf_kwargs = {"tf_topic_namespace": tf_topic_namespace} if tf_topic_namespace else {}
    try:
        # an all-static chain (from_bag_tf needs a dynamic link to get times)
        T = PoseData.static_tf_from_bag(path, parent_frame, child_frame, **tf_kwargs)
        links = "static"
    except AssertionError:
        pose_data = PoseData.from_bag_tf(
            path,
            parent_frame,
            child_frame,
            time_range=[0.0, TRANSFORM_LOOKUP_WINDOW_S],
            time_range_relative=True,
            **tf_kwargs,
        )
        poses = [pose_data.pose(t) for t in pose_data.times]
        T = poses[0]
        for T_i in poses[1:]:
            trans_diff = np.linalg.norm(T_i[:3, 3] - T[:3, 3])
            rot_diff = np.degrees(
                Rot.from_matrix(T[:3, :3].T @ T_i[:3, :3]).magnitude()
            )
            if trans_diff > RIGID_TRANS_TOL_M or rot_diff > RIGID_ROT_TOL_DEG:
                raise ValueError(
                    f"{parent_frame} -> {child_frame} in {path} is not rigid (moved "
                    f"{trans_diff:.4f} m / {rot_diff:.3f} deg in the first "
                    f"{TRANSFORM_LOOKUP_WINDOW_S:.0f} s), so it cannot be looked up as a "
                    "fixed transform. Give it as a 4x4 matrix instead."
                )
        links = f"includes dynamic links, rigid over {len(poses)} poses"
    print(
        f"Looked up {parent_frame} -> {child_frame} from {path} ({links}):\n"
        f"{np.array2string(T, precision=6, suppress_small=True)}"
    )
    return T


def resolve_transform(value, default_path=None):
    """A 4x4 transform given as a matrix (nested or flat list / array) or as a lookup from a
    bag's TF:

        parent_frame: robot/body
        child_frame: robot_zed_left_camera_optical_frame
        path: /path/to/bag        # optional if default_path is given
        tf_topic_namespace: robot  # optional, for /robot/tf and /robot/tf_static

    Raises ValueError (ParamsBase.from_yaml falls back to default params on any other
    exception, which would silently drop the pose data).
    """
    if not isinstance(value, dict):
        return np.array(value, dtype=float).reshape((4, 4))
    unknown = set(value) - TRANSFORM_LOOKUP_KEYS
    if unknown:
        raise ValueError(f"Unknown transform lookup keys {sorted(unknown)} in {value}")
    if "parent_frame" not in value or "child_frame" not in value:
        raise ValueError(
            f"A transform lookup needs parent_frame and child_frame: {value}"
        )
    path = value.get("path", default_path)
    if path is None:
        raise ValueError(
            f"A transform lookup needs a path (the pose data is not from a bag): {value}"
        )
    try:
        T = lookup_bag_transform(
            os.path.expanduser(str(path)),
            value["parent_frame"],
            value["child_frame"],
            value.get("tf_topic_namespace"),
        )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Transform lookup {value} in {path} failed: {e!r}") from e
    return T.copy()


def resolve_pose_data_transforms(pose_data):
    """Resolve the T_premultiply / T_postmultiply of a robotdatapy pose data dict (as passed
    to PoseData.from_dict) to 4x4 matrices (see resolve_transform). Lookups default to the
    pose data's own bag. Returns the dict, updated in place.
    """
    default_path = (
        pose_data.get("path") if pose_data.get("type") in BAG_POSE_DATA_TYPES else None
    )
    for key in POSE_DATA_TRANSFORM_KEYS:
        if pose_data.get(key) is not None:
            pose_data[key] = resolve_transform(pose_data[key], default_path)
    return pose_data
