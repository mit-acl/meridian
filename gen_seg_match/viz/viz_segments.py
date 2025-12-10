import numpy as np
import open3d as o3d
from gen_seg_match.segment.segment_types import SegmentList, SegmentPoint, SegmentLine


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
                z_axis = np.array([0, 0, 1.0])
                v = np.cross(z_axis, seg.get_direction())
                c = np.dot(z_axis, seg.get_direction())
                if np.linalg.norm(v) > 1e-6:
                    vx = np.array(
                        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
                    )
                    R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
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
