import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy import ndimage
from typing import List, Union
from robotdatapy import camera
import robotdatapy as rdp
import cv2 as cv

from gen_seg_match.map3d.observation import Observation
from gen_seg_match.segment.segment_types import (
    SegmentList,
    SegmentPoint,
    SegmentLine,
    SegmentPlane,
    GeneralSegment,
)
from gen_seg_match.segment.map_segment import MapSegment
from gen_seg_match.viz.utils import color_from_seed


def _make_material(shader="defaultUnlit", point_size=None):
    """Create an Open3D MaterialRecord with common defaults."""
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = shader
    if point_size is not None:
        mat.point_size = point_size
    return mat


def viz_segments(
    segments: Union[SegmentList | List[Observation]],
    id_range=None,
    time_range=None,
    time_range_relative=True,
    offscreen=False,
    show_labels=False,
    show_dense=True,
    show_sparse=True,
    colors=None,
    line_radius: float = 0.05,
    point_radius: float = 0.05,
    inf_line_len: float = 10.0,
    plane_size: float = 4.0,
    plane_opacity: float = 0.8,
    dense_opacity: float = 0.8,
    dense_point_size: float = 3.0,
):
    geometry_list = []  # list of (geometry, MaterialRecord) tuples
    label_list = []

    if time_range is not None and time_range_relative:
        time_range = np.array(time_range) + segments.first_seen

    if colors is None:
        colors = [None for _ in range(len(segments))]

    for seg, color in zip(segments, colors):
        if id_range is not None:
            if not (seg.id > id_range[0] and seg.id < id_range[1]):
                continue

        if time_range is not None:
            if seg.first_seen > time_range[1] or seg.last_seen < time_range[0]:
                continue

        # Resolve color once per segment
        if color is None:
            color = seg.color_from_id(num_type=float)

        # ---- dense points ----
        if isinstance(seg, GeneralSegment):
            points = seg.dense_points
        elif isinstance(seg, Observation):
            points = seg.point_cloud
        else:
            points = None

        if show_dense and points is not None:
            num_pts = points.shape[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            color_repeated = np.repeat(np.array(color).reshape((1, 3)), num_pts, axis=0)
            pcd.colors = o3d.utility.Vector3dVector(color_repeated)
            mat = _make_material(
                shader="defaultLitTransparency", point_size=dense_point_size
            )
            mat.base_color = [color[0], color[1], color[2], dense_opacity]
            geometry_list.append((pcd, mat))

        # ---- sparse geometry ----
        if show_sparse:
            if type(seg) is SegmentLine:
                # Determine the two visualized endpoints
                if seg.num_endpoints == 2:
                    pt1, pt2 = seg.endpoints[0], seg.endpoints[1]
                elif seg.num_endpoints == 1:
                    # Ray – keep the existing endpoint and extend along direction
                    ep = (
                        seg.endpoints[0]
                        if seg.endpoints[0] is not None
                        else seg.endpoints[1]
                    )
                    pt1 = ep
                    pt2 = ep + seg.get_direction() * inf_line_len
                else:
                    # No endpoints – extend both ways from the reference point
                    half = inf_line_len / 2.0
                    pt1 = seg.get_point() - half * seg.get_direction()
                    pt2 = seg.get_point() + half * seg.get_direction()

                length = np.linalg.norm(pt2 - pt1)
                midpoint = (pt1 + pt2) / 2.0

                cyl = o3d.geometry.TriangleMesh.create_cylinder(line_radius, length)
                cyl.compute_vertex_normals()

                # Rotate cylinder (default along z) to align with segment direction
                z_axis = np.array([0, 0, 1.0])
                target = seg.get_direction()
                R = Rot.align_vectors([target], [z_axis])[0].as_matrix().T
                cyl.rotate(R)
                cyl.translate(midpoint)
                cyl.paint_uniform_color(color)
                geometry_list.append((cyl, _make_material()))

            elif type(seg) is SegmentPoint:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=point_radius)
                sphere.compute_vertex_normals()
                sphere.translate(seg.get_point())
                sphere.paint_uniform_color(color)
                geometry_list.append((sphere, _make_material()))

            elif type(seg) is SegmentPlane:
                # Build a square patch centred on plane.point, perpendicular to normal
                normal = seg.get_normal()
                ref = (
                    np.array([0.0, 0.0, 1.0])
                    if abs(normal[2]) < 0.9
                    else np.array([1.0, 0.0, 0.0])
                )
                v1 = np.cross(normal, ref)
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 /= np.linalg.norm(v2)

                h = plane_size / 2.0
                corners = np.array(
                    [
                        seg.get_point() - h * v1 - h * v2,
                        seg.get_point() + h * v1 - h * v2,
                        seg.get_point() + h * v1 + h * v2,
                        seg.get_point() - h * v1 + h * v2,
                    ]
                )

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(corners)
                mesh.triangles = o3d.utility.Vector3iVector(
                    np.array([[0, 1, 2], [0, 2, 3]])
                )
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(color)
                mat = _make_material(shader="defaultLitTransparency")
                mat.base_color = [color[0], color[1], color[2], plane_opacity]
                geometry_list.append((mesh, mat))

        # ---- labels ----
        if show_labels:
            label_list.append((seg.get_point(), f"id:{seg.id}", color))

    if not offscreen:
        render3d_onscreen(geometry_list, label_list, segments.get_mean_point())

    return geometry_list, label_list


def render3d_on_img(
    objs_list,
    camera_params: camera.CameraParams,
    camera_pose: np.ndarray = None,
    camera_offset: np.ndarray = None,
    label_list=None,
):
    """Render Open3D geometries to an image.

    Args:
        objs_list: list of (geometry, MaterialRecord) tuples, or plain
            geometries (a default material is used in that case).
        camera_params: camera intrinsics.
        camera_pose: 4x4 extrinsic transform. If None a default is used.
        camera_offset: 3-element translation offset applied on top of
            camera_pose (useful for pulling the camera back so segments
            are more visible).
        label_list: list of (point_3d, text) tuples to overlay as text.
    """
    if camera_pose is None:
        camera_pose = rdp.transform.xyz_rpy_to_transform(
            [
                1.0,
                -2.0,
                -2.0,
            ],
            [
                -10.0,
                -10.0,
                0.0,
            ],
            degrees=True,
        )

    if camera_offset is not None:
        camera_pose = camera_pose.copy()
        camera_pose[:3, 3] += np.asarray(camera_offset)

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        camera_params.width, camera_params.height
    )
    scene = renderer.scene

    renderer.setup_camera(
        camera_params.K,
        np.linalg.inv(camera_pose),
        camera_params.width,
        camera_params.height,
    )
    # Manually widen clipping range
    scene.camera.set_projection(
        camera_params.K,
        0.01,  # near
        100.0,  # far
        camera_params.width,
        camera_params.height,
    )

    for i, item in enumerate(objs_list):
        if isinstance(item, tuple):
            obj, mat = item
        else:
            obj = item
            mat = _make_material(point_size=5.0)
        scene.add_geometry(f"obj-{i}", obj, mat)

    o3d_img = renderer.render_to_image()
    o3d_img = cv.cvtColor(np.asarray(o3d_img), cv.COLOR_RGB2BGR)
    scene.clear_geometry()

    if label_list:
        T_cam = np.linalg.inv(camera_pose)  # world-to-camera
        for pt_3d, text, color in label_list:
            pt_cam = (
                T_cam[:3, :3] @ np.asarray(pt_3d).reshape(3, 1) + T_cam[:3, 3:4]
            ).flatten()
            if pt_cam[2] <= 0:
                continue  # behind camera
            px = camera_params.K @ pt_cam
            u, v = int(px[0] / px[2]), int(px[1] / px[2])
            if 0 <= u < camera_params.width and 0 <= v < camera_params.height:
                cv.putText(
                    o3d_img,
                    text,
                    (u, v),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    tuple(map(lambda c: int(255 * c), color[::-1])),
                    1,
                    cv.LINE_AA,
                )

    return o3d_img


def render3d_onscreen(geometry_list, label_list, mean_point):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)

    for i, item in enumerate(geometry_list):
        if isinstance(item, tuple):
            obj, mat = item
        else:
            obj = item
            mat = _make_material(point_size=5.0)
        vis.add_geometry(f"geom-{i}", obj, mat)
    for label in label_list:
        vis.add_3d_label(*label[:2])

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
    draw_occluded_points=False,
    draw_masks=True,
    cam_params=None,
) -> np.ndarray:
    assert not (draw_points and cam_params is None), (
        "If draw_points is True, cam_params must be provided."
    )

    viz_img = np.array(img_bgr).copy().astype(np.float32)
    for obs in observations:
        if draw_masks:
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

        point_collections_to_draw = []
        point_collections_color = []
        # TODO should fix this but right now this is handling both Observation or MapSegments
        if draw_points and isinstance(obs, Observation) and obs.point_cloud is not None:
            point_collections_to_draw.append(obs.point_cloud)
            point_collections_color.append([0, 0, 0])

        if draw_points and isinstance(obs, MapSegment) and obs.points is not None:
            point_collections_to_draw.append(obs.points)
            point_collections_color.append(color_from_seed(obs.id, num_type="int"))

        if draw_occluded_points and obs.occluded_points is not None:
            point_collections_to_draw.append(obs.occluded_points)
            point_collections_color.append([0, 0, 150])

        for points3d, color in zip(point_collections_to_draw, point_collections_color):
            # Project points to 2D
            points3d_in_front = np.array([p for p in points3d if p[2] > 0])
            points_in_2d = camera.xyz_2_pixel(points3d_in_front, cam_params.K, axis=0)
            for px in points_in_2d:
                px = px.reshape(-1)
                cv.circle(viz_img, (int(px[0]), int(px[1])), 1, color, -1)

    viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
    return viz_img
