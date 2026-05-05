import numpy as np
import cv2 as cv
from typing import List, Tuple

import robotdatapy as rdp

from meridian.primitive.primitive import PointPrimitive, LinePrimitive, Primitive
from meridian.primitive.primitive_list import PrimitiveList


def draw_infinite_line_on_img(
    img,
    line: LinePrimitive,
    K: np.ndarray,
    color=(0, 255, 0),
    thickness=2,
):
    raise Warning(
        "draw_infinite_line_on_img is deprecated; use draw_line_on_img instead."
    )
    unit_vec = line.direction / np.linalg.norm(line.direction)
    points_in_3d = [line.point + unit_vec * i for i in np.linspace(-10, 10, 100)]
    # print(points_in_3d)
    points_in_2d = [
        rdp.camera.xyz_2_pixel(p.reshape((3, 1)), K) for p in points_in_3d if p[2] > 0
    ]
    for i in range(len(points_in_2d) - 1):
        cv.line(
            img,
            tuple(points_in_2d[i].flatten().astype(int)),
            tuple(points_in_2d[i + 1].flatten().astype(int)),
            color,
            thickness,
        )
    return img


def draw_line_on_img(
    img,
    line: LinePrimitive,
    K: np.ndarray,
    color=(0, 255, 0),
    thickness=2,
):
    if line.num_endpoints == 0:
        unit_vec = line.direction / np.linalg.norm(line.direction)
        points_in_3d = [line.point + unit_vec * i for i in np.linspace(-10, 10, 100)]
    elif line.num_endpoints == 1:
        unit_vec = line.direction / np.linalg.norm(line.direction)
        points_in_3d = [line.point + unit_vec * i for i in np.linspace(0, 10, 100)]
    else:  # 2 endpoints
        points_in_3d = [
            line.endpoints[0] + (line.endpoints[1] - line.endpoints[0]) * i
            for i in np.linspace(0, 1, 100)
        ]
    points_in_2d = [
        rdp.camera.xyz_2_pixel(p.reshape((3, 1)), K) for p in points_in_3d if p[2] > 0
    ]
    for i in range(len(points_in_2d) - 1):
        cv.line(
            img,
            tuple(points_in_2d[i].flatten().astype(int)),
            tuple(points_in_2d[i + 1].flatten().astype(int)),
            color,
            thickness,
        )
    return img


def write_seg_id(img, seg, K, color):
    px = rdp.camera.xyz_2_pixel(seg.point.reshape(1, 3), K).reshape(-1)
    cv.putText(
        img,
        str(seg.id),
        (int(px[0]) + 10, int(px[1]) + 10),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )


def img_sparse_viz(
    img: np.ndarray,
    segments: List[Primitive],
    K: np.ndarray,
    colors: List[Tuple[int, int, int]] = None,
    write_ids=False,
):
    img = img.copy()
    segments = PrimitiveList(segments)
    if colors is None:
        colors = [seg.color_from_id(order="bgr") for seg in segments]
    for seg, color in zip(segments, colors):
        if type(seg) == PointPrimitive:
            # TODO: below shouldn't happen
            px = rdp.camera.xyz_2_pixel(seg.point.reshape(1, 3), K).reshape(-1)
            if np.any(np.isnan(px)):
                continue
            cv.circle(img, (int(px[0]), int(px[1])), 15, color, -1)

        elif type(seg) == LinePrimitive:
            img = draw_line_on_img(img, seg, K, color=color, thickness=3)

        if write_ids:
            write_seg_id(img, seg, K, color=color)

    return img
