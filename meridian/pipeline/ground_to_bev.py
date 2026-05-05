import numpy as np
import open3d as o3d
import cv2 as cv
import os
import argparse
import shutil
import tqdm
from dataclasses import dataclass

from meridian.params import GroundToBEVParams, GroundToBEVDataParams
from meridian.pipeline.data import GroundToBEVData


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
    color_aggregation_method: str = "mean",
    color_aggregation_k: int = 5,
    hole_fill_method: str = None,
    hole_fill_radius: int = 5,
) -> np.ndarray:
    """
    Render a bird's eye view image of the point cloud.

    Args:
        pcd: Open3D point cloud.
        resolution: Meters per pixel.
        padding: Padding around the point cloud bounds (meters).
        color_aggregation_method: How to aggregate colors in each cell.
            - "mean": Average all point colors in the cell.
            - "top-1": Use the color of the point with the highest z value.
            - "top-k": Average colors of the k points with highest z values.
        color_aggregation_k: k value for "top-k" aggregation.
        hole_fill_method: Method to fill holes (pixels without points).
            - None: No hole filling.
            - "inpaint": OpenCV inpainting (Telea algorithm).
            - "nearest": Copy from nearest valid pixel.
            - "dilate": Morphological dilation.
        hole_fill_radius: Radius for hole filling operations (pixels).

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

    # Project points to image
    px = ((points[:, 0] - x_min) / resolution).astype(int)
    py = ((points[:, 1] - y_min) / resolution).astype(int)

    # Clip to image bounds
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid]
    py = py[valid]
    valid_colors = colors[valid]
    valid_z = points[valid, 2]

    # Flip Y for image coordinates
    py_flipped = height - 1 - py

    if color_aggregation_method == "mean":
        # Average all point colors in each cell
        bev_img = np.zeros((height, width, 3), dtype=np.float32)
        count_img = np.zeros((height, width), dtype=np.float32)

        for i in range(len(px)):
            bev_img[py_flipped[i], px[i]] += valid_colors[i]
            count_img[py_flipped[i], px[i]] += 1

        valid_mask = count_img > 0
        bev_img[valid_mask] /= count_img[valid_mask, np.newaxis]

    elif color_aggregation_method == "top-1":
        # Use color of the point with highest z value in each cell
        bev_img = np.zeros((height, width, 3), dtype=np.float32)
        z_img = np.full((height, width), -np.inf, dtype=np.float32)

        for i in range(len(px)):
            if valid_z[i] > z_img[py_flipped[i], px[i]]:
                z_img[py_flipped[i], px[i]] = valid_z[i]
                bev_img[py_flipped[i], px[i]] = valid_colors[i]

        valid_mask = z_img > -np.inf

    elif color_aggregation_method == "top-k":
        # Average colors of the k points with highest z values in each cell
        from collections import defaultdict
        import heapq

        # Group points by cell, keeping top-k by z value
        cell_points = defaultdict(list)
        for i in range(len(px)):
            cell = (py_flipped[i], px[i])
            # Use negative z for min-heap (we want max-k)
            if len(cell_points[cell]) < color_aggregation_k:
                heapq.heappush(cell_points[cell], (valid_z[i], valid_colors[i]))
            elif valid_z[i] > cell_points[cell][0][0]:
                heapq.heapreplace(cell_points[cell], (valid_z[i], valid_colors[i]))

        bev_img = np.zeros((height, width, 3), dtype=np.float32)
        valid_mask = np.zeros((height, width), dtype=bool)

        for (row, col), points_list in cell_points.items():
            if points_list:
                colors_arr = np.array([c for _, c in points_list])
                bev_img[row, col] = colors_arr.mean(axis=0)
                valid_mask[row, col] = True

    else:
        raise ValueError(
            f"Unknown color_aggregation_method: {color_aggregation_method}. "
            "Choose from 'mean', 'top-1', or 'top-k'."
        )

    # Convert to uint8
    bev_img = (bev_img * 255).astype(np.uint8)

    # Hole filling
    if hole_fill_method is not None:
        bev_img = fill_holes(
            bev_img, ~valid_mask, method=hole_fill_method, radius=hole_fill_radius
        )

    return bev_img


def fill_holes(
    img: np.ndarray,
    hole_mask: np.ndarray,
    method: str = "inpaint",
    radius: int = 5,
) -> np.ndarray:
    """
    Fill holes in an image.

    Args:
        img: Input image (H, W, 3) in uint8.
        hole_mask: Boolean mask where True indicates holes to fill.
        method: Hole filling method.
            - "inpaint": OpenCV inpainting (Telea algorithm).
            - "nearest": Copy from nearest valid pixel using distance transform.
            - "dilate": Iterative morphological dilation.
        radius: Radius for hole filling operations.

    Returns:
        Image with holes filled.
    """
    if not hole_mask.any():
        return img

    if method == "inpaint":
        # OpenCV inpainting - designed for filling holes naturally
        mask_uint8 = hole_mask.astype(np.uint8) * 255
        result = cv.inpaint(img, mask_uint8, radius, cv.INPAINT_TELEA)

    elif method == "nearest":
        # Use distance transform to find nearest valid pixel
        from scipy import ndimage

        result = img.copy()
        # For each channel, fill holes with nearest valid pixel
        for c in range(3):
            channel = img[:, :, c].astype(np.float32)
            # Distance transform gives distance to nearest non-hole pixel
            # and indices gives the coordinates of that pixel
            _, indices = ndimage.distance_transform_edt(
                hole_mask, return_distances=True, return_indices=True
            )
            result[:, :, c] = channel[indices[0], indices[1]]

    elif method == "dilate":
        # Iterative dilation to fill holes
        result = img.copy()
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        remaining_holes = hole_mask.copy()

        for _ in range(radius * 2):  # Multiple iterations to fill larger holes
            if not remaining_holes.any():
                break

            # Dilate valid regions into holes
            for c in range(3):
                dilated = cv.dilate(result[:, :, c], kernel)
                result[:, :, c] = np.where(remaining_holes, dilated, result[:, :, c])

            # Update remaining holes (pixels that were holes and still are black)
            still_black = result.sum(axis=2) == 0
            remaining_holes = remaining_holes & still_black

    else:
        raise ValueError(
            f"Unknown hole_fill_method: {method}. "
            "Choose from 'inpaint', 'nearest', or 'dilate'."
        )

    return result


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
            if (
                last_position is None
                or np.linalg.norm(curr_position - last_position)
                > self.params.sample_distance
            ):
                last_position = curr_position
                times.append(t)
            img_idx += 1

        print(f"Processing {len(times)} images from t={t0:.2f} to t={tf:.2f}")

        if len(times) == 0:
            print(
                "[WARNING] No images to process! Check time range and sample_distance."
            )
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
        bev_img = render_bev_image(
            pcd,
            resolution=self.params.bev_resolution,
            color_aggregation_method=self.params.color_aggregation_method,
            color_aggregation_k=self.params.color_aggregation_k,
            hole_fill_method=self.params.hole_fill_method,
            hole_fill_radius=self.params.hole_fill_radius,
        )
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
        shutil.copy2(
            params_path, os.path.join(output_dir, os.path.basename(params_path))
        )
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
