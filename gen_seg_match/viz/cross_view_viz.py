import numpy as np
import matplotlib.pyplot as plt

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint, SegmentList
from gen_seg_match.viz.utils import color_from_seed

color_list = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def plot_seg(seg, ax, custom_color=None):
    color = (
        seg.color_from_id(order="rgb", num_type="float")
        if custom_color is None
        else custom_color
    )
    if isinstance(seg, SegmentLine):
        assert seg.num_endpoints == 2, (
            "only supports line segments currently (no infinite lines)"
        )
        ax.plot(
            [seg.endpoints[0][0], seg.endpoints[1][0]],
            [seg.endpoints[0][1], seg.endpoints[1][1]],
            color=color,
            linewidth=4,
        )
    elif isinstance(seg, SegmentPoint):
        ax.plot(
            seg.get_point()[0],
            seg.get_point()[1],
            "o",
            color=color,
            markersize=5,
            markeredgewidth=3,
        )


def viz_cross_view_matches(
    aerial_segments: SegmentList, ground_segments: SegmentList, matches: np.ndarray
):
    color = "k"
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].set_title("Aerial Segments")
    for line in aerial_segments.get_lines():
        plot_seg(line, ax[0], custom_color=color)
    for point in aerial_segments.get_points():
        plot_seg(point, ax[0], custom_color=color)
    ax[0].axis("equal")
    ax[0].invert_yaxis()

    ax[1].axis("equal")
    for line in ground_segments.get_lines():
        plot_seg(line, ax[1], custom_color=color)
    for point in ground_segments.get_points():
        plot_seg(point, ax[1], custom_color=color)
    ax[1].set_title("Ground Segments")

    for i, match in enumerate(matches):
        seg1 = ground_segments.get_segment_from_id(match[0])
        seg2 = aerial_segments.get_segment_from_id(match[1])
        random_color = color_from_seed(i, order="rgb", num_type="float")
        plot_seg(seg1, ax[1], custom_color=random_color)
        plot_seg(seg2, ax[0], custom_color=random_color)

    [ax[i].grid(True) for i in range(2)]
    plt.show()
