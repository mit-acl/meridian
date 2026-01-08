import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy import ndimage
from typing import List
from robotdatapy import camera
import cv2 as cv

from roman.map.observation import Observation
from gen_seg_match.segment.segment_types import SegmentList, SegmentPoint, SegmentLine
from gen_seg_match.viz.utils import color_from_seed


def viz_segments(
    segments: SegmentList,
    id_range=None,
    time_range=None,
    time_range_relative=True,
    offscreen=False,
    show_labels=False,
    show_dense=True,
    show_sparse=True,
):
    geometry_list = []
    label_list = []

    # TODO: do we want to support time ranges?
    if time_range is not None and time_range_relative:
        time_range = np.array(time_range) + segments.first_seen

    for seg in segments:
        if id_range is not None:
            if not (seg.id > id_range[0] and seg.id < id_range[1]):
                continue

        if time_range is not None:
            if seg.first_seen > time_range[1] or seg.last_seen < time_range[0]:
                continue

        if show_dense and seg.dense_points is not None:
            num_pts = seg.dense_points.shape[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg.dense_points)
            color = np.repeat(
                np.array(seg.color_from_id(num_type=float)).reshape((1, 3)),
                num_pts,
                axis=0,
            )
            pcd.colors = o3d.utility.Vector3dVector(color)
            geometry_list.append(pcd)

        if show_sparse:
            if type(seg) is SegmentLine:
                cyl = o3d.geometry.TriangleMesh.create_cylinder(0.05, seg.get_length())
                cyl.compute_vertex_normals()

                # Compute rotation
                # Find some valid rotation matrix that aligns z axis to segment direction
                z_axis = np.array([0, 0, 1.0])
                target = seg.get_direction()
                R = Rot.align_vectors([target], [z_axis]).as_matrix()[0].T
                cyl.rotate(R)

                # Move to midpoint
                cyl.translate((seg.endpoints[0] + seg.endpoints[1]) / 2)
                cyl.paint_uniform_color(seg.color_from_id(num_type=float))
                geometry_list.append(cyl)
            elif type(seg) is SegmentPoint:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.compute_vertex_normals()
                sphere.translate(seg.get_point())
                sphere.paint_uniform_color(seg.color_from_id(num_type=float))
                geometry_list.append(sphere)

        if show_labels:
            label = [f"id: {seg.id}"]
            for i in range(len(label)):
                label_list.append(
                    (seg.get_point() + np.array([0, 0, -0.15 * i]), label[i])
                )

    if not offscreen:
        render3d_onscreen(geometry_list, label_list, segments.get_mean_point())
    else:
        return geometry_list, label_list


def render3d_onscreen(geometry_list, label_list, mean_point):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5.0

    for i, obj in enumerate(geometry_list):
        vis.add_geometry(f"geom-{i}", obj, mat)
    for label in label_list:
        vis.add_3d_label(*label)

    K = np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]]).astype(np.float64)
    T_inv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 20], [0, 0, 0, 1]]
    ).astype(np.float64)
    T_inv[:3, 3] += mean_point
    T = np.linalg.inv(T_inv)
    vis.setup_camera(K, T, 400, 400)
    app.add_window(vis)
    app.run()


def viz_masks_on_img(
    img_bgr: np.ndarray,
    observations: List[Observation],
    alpha: float = 0.5,
    colors=None,
    draw_points=False,
    cam_params=None,
) -> np.ndarray:
    assert not (draw_points and cam_params is None), (
        "If draw_points is True, cam_params must be provided."
    )

    viz_img = np.array(img_bgr).copy().astype(np.float32)
    for obs in observations:
        mask = obs.mask.astype(bool)
        if colors is not None:
            color = np.array(colors[obs.id % len(colors)], dtype=np.float32)
        else:
            color = np.array(
                color_from_seed(np.random.randint(0, 1000)), dtype=np.float32
            )

        viz_img[mask] = (1 - alpha) * viz_img[mask] + alpha * color
        edges = ndimage.binary_dilation(mask) ^ mask
        viz_img[edges] = 0

        if draw_points and obs.point_cloud is not None:
            # Project points to 2D
            points_in_2d = [
                camera.xyz_2_pixel(p.reshape((3, 1)), cam_params.K)
                for p in obs.point_cloud
                if p[2] > 0
            ]
            for px in points_in_2d:
                px = px.reshape(-1)
                cv.circle(viz_img, (int(px[0]), int(px[1])), 1, [0, 0, 0], -1)

    viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
    return viz_img
