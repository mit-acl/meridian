"""Render a 2D top-down view of a SegmentMap for quick inspection."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from meridian.map3d.map import SegmentMap
from meridian.utils import clean_up_points
from meridian.viz.utils import color_from_seed


def visualize_segment_map(
    input_path: str,
    output_path: str = None,
    voxel_size: float = 0.5,
):
    seg_map = SegmentMap.from_pickle(input_path)

    fig, ax = plt.subplots(figsize=(12, 12))

    if seg_map.trajectory:
        traj = np.array([T[:3, 3] for T in seg_map.trajectory])
        ax.plot(traj[:, 0], traj[:, 1], "k-", linewidth=1.0, alpha=0.5)

    n_drawn = 0
    for seg in seg_map.segments:
        if seg.points is None or len(seg.points) == 0:
            continue
        pts = seg.points
        if voxel_size is not None and voxel_size > 0:
            pts = clean_up_points(pts, voxel_size=voxel_size)
            if pts is None or len(pts) == 0:
                continue
        color = color_from_seed(seg.id, order="rgb", num_type="float")
        ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, color=color, alpha=0.8)
        n_drawn += 1

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"SegmentMap: {n_drawn}/{len(seg_map.segments)} segments drawn, "
        f"{len(seg_map.times)} poses, voxel_size={voxel_size}m"
    )
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Path to segment_map.pkl")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path (PNG); if omitted, shows interactively",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
        help="Voxel-grid downsample size in meters (default 0.5; set 0 to disable)",
    )
    args = parser.parse_args()

    visualize_segment_map(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
    )


if __name__ == "__main__":
    main()
