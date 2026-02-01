import numpy as np
import open3d as o3d
import cv2 as cv
import os
import argparse
import shutil
import tqdm
from dataclasses import dataclass

from gen_seg_match.params import GroundToBEVParams, GroundToBEVDataParams
from gen_seg_match.pipeline.data import GroundToBEVData


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    max_depth: float,
    depth_scale: float = 1.0,
) -> o3d.geometry.PointCloud:
    """
    Convert an RGBD image to a colored Open3D point cloud.

    Args:
        rgb: RGB image (H, W, 3) in uint8.
        depth: Depth image (H, W).
        intrinsics: Open3D camera intrinsics.
        max_depth: Maximum depth to include (meters).
        depth_scale: Multiplier to convert depth values to meters.

    Returns:
        Open3D PointCloud with colors.
    """
    # Convert to Open3D images
    # Note: Open3D expects RGB, not BGR
    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale,  # Open3D uses divisor, we use multiplier
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsics,
        project_valid_depth_only=True,
    )

    return pcd


def render_bev_image(
    pcd: o3d.geometry.PointCloud,
    resolution: float = 0.02,
    padding: float = 1.0,
) -> np.ndarray:
    """
    Render a bird's eye view image of the point cloud.

    Args:
        pcd: Open3D point cloud.
        resolution: Meters per pixel.
        padding: Padding around the point cloud bounds (meters).

    Returns:
        BEV image as numpy array (H, W, 3) in uint8.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Get XY bounds (assuming Z is up/down)
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding

    # Calculate image dimensions
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)

    # Create empty image
    bev_img = np.zeros((height, width, 3), dtype=np.float32)
    count_img = np.zeros((height, width), dtype=np.float32)

    # Project points to image
    px = ((points[:, 0] - x_min) / resolution).astype(int)
    py = ((points[:, 1] - y_min) / resolution).astype(int)

    # Clip to image bounds
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid]
    py = py[valid]
    valid_colors = colors[valid]

    # Accumulate colors (flip Y for image coordinates)
    py_flipped = height - 1 - py
    for i in range(len(px)):
        bev_img[py_flipped[i], px[i]] += valid_colors[i]
        count_img[py_flipped[i], px[i]] += 1

    # Average colors where we have points
    mask = count_img > 0
    bev_img[mask] /= count_img[mask, np.newaxis]

    # Convert to uint8
    bev_img = (bev_img * 255).astype(np.uint8)

    return bev_img


@dataclass
class GroundToBEV:
    params: GroundToBEVParams

    def build_pointcloud(self, data: GroundToBEVData) -> o3d.geometry.PointCloud:
        """
        Build a merged point cloud from RGBD images.

        Args:
            data: GroundToBEVData containing image, depth, and pose data.

        Returns:
            Merged Open3D point cloud in world frame.
        """
        # Determine time range from loaded data (already filtered during loading)
        t0 = max(data.img_data.t0, data.depth_data.t0, data.camera_pose_data.t0)
        tf = min(data.img_data.tf, data.depth_data.tf, data.camera_pose_data.tf)

        # Sample image times based on distance traveled
        times = []
        last_position = None
        img_idx = data.img_data.idx(t0, force_single=True)
        img_idx_tf = data.img_data.idx(tf, force_single=True)

        while img_idx < img_idx_tf:
            t = data.img_data.times[img_idx]
            curr_position = data.camera_pose_data.position(t)
            if last_position is None or np.linalg.norm(
                curr_position - last_position
            ) > self.params.sample_distance:
                last_position = curr_position
                times.append(t)
            img_idx += 1

        print(f"Processing {len(times)} images from t={t0:.2f} to t={tf:.2f}")

        if len(times) == 0:
            print("[WARNING] No images to process! Check time range and sample_distance.")
            return o3d.geometry.PointCloud()

        # Create camera intrinsics for Open3D
        cam_params = data.img_data.camera_params
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=cam_params.width,
            height=cam_params.height,
            fx=cam_params.fx,
            fy=cam_params.fy,
            cx=cam_params.cx,
            cy=cam_params.cy,
        )

        # Build merged point cloud
        merged_pcd = o3d.geometry.PointCloud()

        for t in tqdm.tqdm(times, desc="Building point cloud"):
            # Get RGB, depth, and pose at time t
            rgb = data.img_data.img(t)
            depth = data.depth_data.img(t)
            T_world_cam = data.camera_pose_data.pose(t)

            # Convert BGR to RGB for Open3D
            rgb_rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)

            # Convert to point cloud in camera frame
            pcd_cam = rgbd_to_pointcloud(
                rgb=rgb_rgb,
                depth=depth,
                intrinsics=intrinsics,
                max_depth=self.params.max_depth,
                depth_scale=self.params.depth_scale,
            )

            # Transform to world frame
            pcd_cam.transform(T_world_cam)

            # Merge with accumulated point cloud
            merged_pcd += pcd_cam

            # Voxel downsample periodically to manage memory
            if len(merged_pcd.points) > 1000000:
                merged_pcd = merged_pcd.voxel_down_sample(self.params.voxel_size)

        # Final voxel downsample
        merged_pcd = merged_pcd.voxel_down_sample(self.params.voxel_size)

        print(f"Final point cloud has {len(merged_pcd.points)} points")

        return merged_pcd

    def run(self, data: GroundToBEVData, output_dir: str):
        """
        Build point cloud and save outputs.

        Args:
            data: GroundToBEVData containing image, depth, and pose data.
            output_dir: Directory to save outputs.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build point cloud
        pcd = self.build_pointcloud(data)

        # Save point cloud
        pcd_path = os.path.join(output_dir, "pointcloud.pcd")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Saved point cloud to {pcd_path}")

        # Render and save BEV image
        bev_img = render_bev_image(pcd, resolution=self.params.bev_resolution)
        bev_path = os.path.join(output_dir, "bev.png")
        cv.imwrite(bev_path, cv.cvtColor(bev_img, cv.COLOR_RGB2BGR))
        print(f"Saved BEV image to {bev_path}")


def ground_to_bev(params_path: str, output_dir: str):
    """Run ground to BEV pipeline."""
    print("Loading parameters...")
    pipeline_params = GroundToBEVParams.load(params_path)
    data_params = GroundToBEVDataParams.load(params_path)

    # Compute time range to only load required portion of bags
    time_range = None
    print("Determining bag time range...")
    bag_t_range = GroundToBEVData.get_bag_time_range(data_params)
    if bag_t_range is not None:
        bag_t0, bag_tf = bag_t_range
        t0 = bag_t0 + pipeline_params.start_time
        tf = (
            bag_t0 + pipeline_params.end_time
            if pipeline_params.end_time is not None
            else bag_tf
        )
        tf = min(tf, bag_tf)
        time_range = (t0, tf)
        print(
            f"Loading bag data from t={t0:.2f} to t={tf:.2f} "
            f"(bag range: {bag_t0:.2f} to {bag_tf:.2f})"
        )

    print("Loading data...")
    data = GroundToBEVData.from_params(data_params, time_range=time_range)
    print("Data loaded.")

    runner = GroundToBEV(params=pipeline_params)
    runner.run(data, output_dir)

    # Copy params to output dir
    if os.path.isfile(params_path):
        shutil.copy2(params_path, os.path.join(output_dir, os.path.basename(params_path)))
    else:
        shutil.copytree(params_path, output_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a bird's eye view point cloud from ground RGBD images."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to params directory or file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    args = parser.parse_args()

    ground_to_bev(args.params, args.output)
