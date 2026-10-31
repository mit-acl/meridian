"""Checks that span more than one params class, or params vs. command-line args.

Single-field normalization and validation (lowercasing, expandvars, allowed
values, legacy names) stays in each params class's __post_init__; this module is
for consistency between classes, so the params files stay mostly lists of
names and values.
"""

import sys


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
