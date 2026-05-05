import argparse
import open3d as o3d


def visualize_pointcloud(pcd_path: str):
    """
    Visualize an Open3D point cloud file.

    Args:
        pcd_path: Path to the .pcd or .ply file.
    """
    print(f"Loading point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)

    print(f"Point cloud has {len(pcd.points)} points")

    if pcd.has_colors():
        print("Point cloud has colors")
    else:
        print("Point cloud has no colors, using uniform color")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Print bounds
    points = pcd.points
    if len(points) > 0:
        import numpy as np

        pts = np.asarray(points)
        print(f"X range: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]")
        print(f"Y range: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")
        print(f"Z range: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]")

    # Visualize
    print("Visualizing... (press Q to close)")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Viewer",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an Open3D point cloud.")
    parser.add_argument("pcd_path", type=str, help="Path to .pcd or .ply file")
    args = parser.parse_args()

    visualize_pointcloud(args.pcd_path)
