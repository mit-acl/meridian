import robotdatapy.camera as rdpc
import cv2 as cv
import numpy as np
from meridian.primitive.primitive import PointPrimitive, LinePrimitive


def draw_infinite_line_on_img(
    img,
    line: LinePrimitive,
    camera_params: rdpc.CameraParams,
    color=(0, 255, 0),
    thickness=2,
):
    unit_vec = line.direction / np.linalg.norm(line.direction)
    points_in_3d = [line.point + unit_vec * i for i in np.linspace(-10, 10, 100)]
    # print(points_in_3d)
    points_in_2d = [
        rdpc.xyz_2_pixel(p.reshape((3, 1)), camera_params.K)
        for p in points_in_3d
        if p[2] > 0
    ]
    # print(points_in_2d)
    for p in points_in_3d:
        if p[2] < 0:
            continue
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
    camera_params: rdpc.CameraParams,
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
        rdpc.xyz_2_pixel(p.reshape((3, 1)), camera_params.K)
        for p in points_in_3d
        if p[2] > 0
    ]
    for p in points_in_3d:
        if p[2] < 0:
            continue
    for i in range(len(points_in_2d) - 1):
        cv.line(
            img,
            tuple(points_in_2d[i].flatten().astype(int)),
            tuple(points_in_2d[i + 1].flatten().astype(int)),
            color,
            thickness,
        )
    return img


def draw_segment_types_on_img(
    img, segments, camera_params: rdpc.CameraParams, write_ids=False
):
    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (255, 165, 0),
        (0, 255, 255),
        (255, 255, 0),
        (128, 0, 128),
    ]
    for i, seg in enumerate(segments):
        if type(seg) is PointPrimitive:
            color = colors[i % len(colors)]
            px = rdpc.xyz_2_pixel(seg.point.reshape((3, 1)), camera_params.K).reshape(
                -1
            )
            cv.circle(img, (int(px[0]), int(px[1])), 15, color, -1)
            if write_ids:
                cv.putText(
                    img,
                    str(seg.id),
                    (int(px[0]) + 10, int(px[1]) + 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
        elif type(seg) is LinePrimitive:
            color = colors[i % len(colors)]
            img = draw_line_on_img(img, seg, camera_params, color=color, thickness=3)
            if write_ids:
                px = rdpc.xyz_2_pixel(
                    seg.point.reshape((3, 1)), camera_params.K
                ).reshape(-1)
                cv.putText(
                    img,
                    str(seg.id),
                    (int(px[0]) + 10, int(px[1]) + 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
    return img
