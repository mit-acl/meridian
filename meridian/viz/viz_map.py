import numpy as np
import cv2 as cv
import open3d as o3d

from meridian.map3d.map import SegmentMap


def render_segment_map_image(
    segment_map: SegmentMap,
    width: int = 1920,
    height: int = 1080,
    show_poses: bool = True,
    time_range: tuple = None,
    min_pose_dist: float = 0.5,
) -> np.ndarray:
    """
    Render a 3D visualization of a SegmentMap to an image.

    Args:
        segment_map: SegmentMap to visualize.
        width: Image width in pixels.
        height: Image height in pixels.
        show_poses: Whether to show trajectory poses as coordinate frames.
        time_range: Optional (t0, tf) to filter segments and poses.
        min_pose_dist: Minimum distance between displayed poses.

    Returns:
        BGR image as numpy array (H, W, 3).
    """
    pcd_list = []
    poses_list = []

    # Build point clouds for segments
    for seg in segment_map.segments:
        if time_range is not None:
            if seg.first_seen > time_range[1] or seg.last_seen < time_range[0]:
                continue
        seg_points = seg.points
        if seg_points is not None and seg_points.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg_points)
            color = np.array(seg.viz_color).reshape(1, 3) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(
                np.repeat(color, seg_points.shape[0], axis=0)
            )
            pcd_list.append(pcd)

    # Build pose meshes for trajectory
    if show_poses:
        displayed_positions = []
        for i, pose in enumerate(segment_map.trajectory):
            if time_range is not None:
                t = segment_map.times[i]
                if t < time_range[0] or t > time_range[1]:
                    continue
            if (
                displayed_positions
                and np.linalg.norm(pose[:3, 3] - np.array(displayed_positions[-1]))
                < min_pose_dist
            ):
                continue
            displayed_positions.append(pose[:3, 3])
            pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            pose_obj.transform(pose)
            poses_list.append(pose_obj)

    if len(pcd_list) == 0 and len(poses_list) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Compute camera viewpoint: top-down looking at center of trajectory
    if len(segment_map.trajectory) > 0:
        positions = np.array([p[:3, 3] for p in segment_map.trajectory])
        center = np.mean(positions, axis=0)
        extent = np.max(np.ptp(positions, axis=0))
        camera_height = max(extent * 1.5, 10.0)
    else:
        center = np.zeros(3)
        camera_height = 20.0

    # Camera looking down (Z-up world assumed)
    camera_pos = center + np.array([0.0, 0.0, camera_height])
    eye = camera_pos
    at = center
    up = np.array([1.0, 0.0, 0.0])

    # Render using offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    pt_mat = o3d.visualization.rendering.MaterialRecord()
    pt_mat.point_size = 5.0

    pose_mat = o3d.visualization.rendering.MaterialRecord()

    for i, pcd in enumerate(pcd_list):
        scene.add_geometry(f"pcd-{i}", pcd, pt_mat)

    if show_poses:
        for i, pose_obj in enumerate(poses_list):
            scene.add_geometry(f"pose-{i}", pose_obj, pose_mat)

    renderer.setup_camera(60.0, center, eye, up)

    o3d_img = renderer.render_to_image()
    img_bgr = cv.cvtColor(np.asarray(o3d_img), cv.COLOR_RGB2BGR)
    scene.clear_geometry()

    return img_bgr
