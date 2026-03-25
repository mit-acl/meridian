#########################################
#
# segmenter.py
#
# A Python wrapper for sending RGBD images to a segmentation model and using segmentation
# masks to create object observations.
#
# Adapted from ROMAN's FastSAMWrapper
#
# Authors: Jouko Kinnari, Mason Peterson, Lucas Jia, Annika Thomas, Qingyuan Li
#
# Jan. 7, 2026
#
#########################################


import cv2 as cv
import numpy as np
import open3d as o3d
import copy
import torch
from yolov7_package import Yolov7Detector
from PIL import Image
import clip
from typing import List
from scipy.spatial import cKDTree


from robotdatapy.camera import CameraParams, pixel_depth_2_xyz

from gen_seg_match.params import SegmenterParams
from gen_seg_match.viz.viz_segments import viz_masks_on_img
from gen_seg_match.map3d.observation import Observation
from gen_seg_match.segmenter.segmenter_base import SegmenterBase


class Segmenter(SegmenterBase):
    def __init__(self, params: SegmenterParams, depth_cam_params: CameraParams = None):
        # parameters
        self.depth_cam_params = depth_cam_params

        # member variables
        self.observations = []

        # Initialize base (segmentation model + semantics model)
        super().__init__(params)

        # Check for valid params
        assert self.params.device == "cuda" or self.params.device == "cpu", (
            "Device should be 'cuda' or 'cpu'."
        )
        assert (
            self.params.rotate_img is None
            or self.params.rotate_img == "CW"
            or self.params.rotate_img == "CCW"
            or self.params.rotate_img == "180"
        ), "Invalid rotate_img option."

        # Set up YOLOv7 detector for ignore/keep masks
        assert (
            not self.params.use_keep_labels
            or self.params.keep_labels_option == "intersect"
            or self.params.keep_labels_option == "contain"
        ), "Keep labels option should be one of: intersect, contain"
        if len(self.params.ignore_labels) > 0 or self.params.use_keep_labels:
            self.yolov7_det = Yolov7Detector(
                traced=False,
                img_size=self.params.yolo_imgsz,
                weights=self.params.yolo_weights_path,
            )

        # TODO: do we wawnt some pixel area bounds?
        # TODO: what is keep mask minimal intersection?
        self.run_yolo = (
            len(self.params.ignore_labels) > 0 or self.params.use_keep_labels
        )

        # Set up ignore mask from triangle ignore masks
        if self.params.triangle_ignore_masks is not None:
            self.constant_ignore_mask = np.zeros(
                (self.depth_cam_params.height, self.depth_cam_params.width),
                dtype=np.uint8,
            )
            for triangle in self.params.triangle_ignore_masks:
                assert len(triangle) == 3, "Triangle must have 3 points."
                for pt in triangle:
                    assert len(pt) == 2, "Each point must have 2 coordinates."
                    assert all([isinstance(x, int) for x in pt]), (
                        "Coordinates must be integers."
                    )
                cv.fillPoly(self.constant_ignore_mask, [np.array(triangle)], 1)
            self.constant_ignore_mask = self.apply_rotation(self.constant_ignore_mask)
        else:
            self.constant_ignore_mask = None

        # Set up camera parameters
        if not self.params.use_point_cloud and self.depth_cam_params is not None:
            self.open3d_cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=int(self.depth_cam_params.width),
                height=int(self.depth_cam_params.height),
                fx=self.depth_cam_params.fx,
                fy=self.depth_cam_params.fy,
                cx=self.depth_cam_params.cx,
                cy=self.depth_cam_params.cy,
            )
        if self.params.erosion_size > 0:
            # see: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
            erosion_shape = cv.MORPH_ELLIPSE
            self.erosion_element = cv.getStructuringElement(
                erosion_shape,
                (2 * self.params.erosion_size + 1, 2 * self.params.erosion_size + 1),
                (self.params.erosion_size, self.params.erosion_size),
            )
        else:
            self.erosion_element = None

    def _init_semantics_model(self):
        """Override to add CLIP support on top of base DINO/DINOv3 support."""
        if (
            self.params.semantics is not None
            and self.params.semantics.lower() == "clip"
        ):
            clip_model = "ViT-L/14"
            self.semantics_model, self.semantics_preprocess = clip.load(
                clip_model, device=self.params.device
            )
        else:
            super()._init_semantics_model()

    def set_depth_camera_params(self, depth_cam_params: CameraParams):
        self.depth_cam_params = depth_cam_params
        self.open3d_cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.depth_cam_params.width),
            height=int(self.depth_cam_params.height),
            fx=self.depth_cam_params.fx,
            fy=self.depth_cam_params.fy,
            cx=self.depth_cam_params.cx,
            cy=self.depth_cam_params.cy,
        )

    def segment(self, img_bgr, t=None, pose=None, depth_data=None):
        """
        Takes and image and returns filtered segment masks as Observations.

        Args:
            img (cv image): camera image

        Returns:
            self.observations (list): list of Observations
            frame_descriptor (np.ndarray): semantic descriptor of the frame if frame_descriptor is not None, else None
        """
        if depth_data is not None:
            assert self.depth_cam_params is not None or self.params.use_point_cloud, (
                "Depth camera parameters must be provided if depth data is used."
            )

        self.observations = []

        # rotate image
        img_orig = img_bgr
        img_bgr = self.apply_rotation(img_bgr)

        if self.params.use_point_cloud:
            pcl, pcl_proj = depth_data
            # Build a sparse depth image from the point cloud projections
            # so occlusion edge logic can index it like a dense depth image.
            h, w = img_bgr.shape[:2]
            sparse_depth_img = np.zeros((h, w), dtype=np.float32)
            valid = (
                (pcl_proj[:, 0] >= 0)
                & (pcl_proj[:, 0] < w)
                & (pcl_proj[:, 1] >= 0)
                & (pcl_proj[:, 1] < h)
            )
            sparse_depth_img[pcl_proj[valid, 1], pcl_proj[valid, 0]] = (
                pcl[valid, 2] * self.params.depth_scale
            )

        if self.run_yolo:
            ignore_mask, keep_mask = self._create_mask(img_bgr)
        else:
            ignore_mask = None
            keep_mask = None

        if self.constant_ignore_mask is not None:
            ignore_mask = (
                np.bitwise_or(ignore_mask, self.constant_ignore_mask)
                if ignore_mask is not None
                else self.constant_ignore_mask
            )

        # run segmentation
        masks = self._process_img(img_bgr, ignore_mask=ignore_mask, keep_mask=keep_mask)

        if self.params.semantics in ("dino", "dinov3", "dinov3-hf"):
            dino_features, dino_output_patches = self._extract_dino_features(img_bgr)
            dino_features = self.unapply_rotation(dino_features)
        elif self.params.semantics == "clip":
            dino_output_patches = None
            dino_features = None

        frame_descriptor = None
        if self.frame_descriptor_type is not None and dino_output_patches is not None:
            frame_descriptor = self.get_frame_descriptor(dino_output_patches, img_bgr)

        if depth_data is not None:
            if self.params.use_point_cloud:
                occlusion_edge_mask = self._get_border_occlusion_edge_mask(
                    img_bgr.shape[:2]
                )
            else:
                occlusion_edge_mask = self._get_occlusion_edge_mask(depth_data)

        for mask in masks:
            mask = self.unapply_rotation(mask)
            points = None
            semantic_descriptor = None

            # Extract point cloud of object from RGBD
            if depth_data is not None:
                if self.params.use_point_cloud:
                    # get 3D points that project within the mask
                    inside_mask = mask[pcl_proj[:, 1], pcl_proj[:, 0]] == 1
                    inside_mask_points = pcl[inside_mask]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(inside_mask_points)

                else:
                    depth_obj = copy.deepcopy(depth_data)
                    if self.erosion_element is not None:
                        eroded_mask = cv.erode(mask, self.erosion_element)
                        depth_obj[eroded_mask == 0] = 0
                    else:
                        depth_obj[mask == 0] = 0

                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d.geometry.Image(
                            np.ascontiguousarray(depth_obj).astype(
                                np.dtype(depth_obj.dtype).type
                            )
                        ),
                        self.open3d_cam_intrinsics,
                        depth_scale=self.params.depth_scale,
                        # depth_trunc=self.params.max_depth,
                        stride=self.params.pcd_stride,
                        project_valid_depth_only=True,
                    )

                # shared for depth & rangesens, once PointCloud object is created

                pcd.remove_non_finite_points()
                pcd_sampled = pcd.voxel_down_sample(voxel_size=self.params.voxel_size)
                original_points = np.asarray(pcd_sampled.points)
                points_in_depth_range = original_points[
                    original_points[:, 2] < self.params.max_depth
                ]

                pcd_in_depth_range = o3d.geometry.PointCloud()
                pcd_in_depth_range.points = o3d.utility.Vector3dVector(
                    points_in_depth_range
                )

                if not pcd_in_depth_range.is_empty():
                    points = self._remove_point_cloud_outliers(pcd_in_depth_range)
                if points is None or points.size == 0:
                    continue

            # Generate downsampled mask
            mask_downsampled = np.array(
                cv.resize(
                    mask,
                    (
                        mask.shape[1] // self.params.mask_downsample_factor,
                        mask.shape[0] // self.params.mask_downsample_factor,
                    ),
                    interpolation=cv.INTER_NEAREST,
                )
            ).astype("uint8")

            if self.params.semantics == "clip":
                ### Use bounding box
                bbox = self.mask_bounding_box(mask.astype("uint8"))
                min_col, min_row, max_col, max_row = bbox
                img_bbox = self.apply_rotation(
                    img_orig[min_row:max_row, min_col:max_col]
                )
                img_bbox = cv.cvtColor(img_bbox, cv.COLOR_BGR2RGB)
                processed_img = self.semantics_preprocess(
                    Image.fromarray(img_bbox, mode="RGB")
                ).to(self.params.device)
                clip_embedding = self.semantics_model.encode_image(
                    processed_img.unsqueeze(dim=0)
                )
                clip_embedding = clip_embedding.squeeze().cpu().detach().numpy()
                semantic_descriptor = clip_embedding
            elif self.params.semantics in ("dino", "dinov3", "dinov3-hf"):
                assert (
                    mask.shape[0] == dino_features.shape[0]
                    and mask.shape[1] == dino_features.shape[1]
                ), "Mask and DINO features must have the same shape."
                semantic_descriptor = self._compute_mean_dino_descriptor(
                    dino_features, mask
                )

            new_observation = Observation(
                id=len(self.observations),
                time=t,
                pose=pose,
                mask=mask,
                mask_downsampled=mask_downsampled,
                semantic_descriptor=semantic_descriptor,
            )

            if depth_data is not None:
                new_observation.point_cloud = points
                new_observation.points_beyond_max_depth = original_points[
                    original_points[:, 2] > self.params.max_depth
                ]
                new_observation.occluded_points = self._compute_occlusion_points(
                    points=points,
                    mask=mask,
                    depth_img=sparse_depth_img if self.params.use_point_cloud else depth_data,
                    occlusion_edge_mask=occlusion_edge_mask,
                )

                # Remove non-occluded points too close to occluded points
                if (
                    new_observation.occluded_points is not None
                    and new_observation.occluded_points.shape[0] > 0
                    and new_observation.point_cloud is not None
                    and new_observation.point_cloud.shape[0] > 0
                ):
                    occ_tree = cKDTree(new_observation.occluded_points)
                    dists, _ = occ_tree.query(new_observation.point_cloud)
                    keep = dists > self.params.min_occluded_unoccluded_dist_m
                    new_observation.point_cloud = new_observation.point_cloud[keep]

                if len(new_observation.point_cloud) == 0:
                    continue

            self.observations.append(new_observation)

        return self.observations, frame_descriptor

    def visualize_segments(
        self, img_bgr: np.ndarray, observations: List[Observation], alpha: float = 0.5
    ) -> np.ndarray:
        return viz_masks_on_img(img_bgr, observations, alpha)

    def apply_rotation(self, img, unrotate=False):
        if self.params.rotate_img is None:
            return img
        elif self.params.rotate_img == "CW":
            k = 3 if not unrotate else 1
        elif self.params.rotate_img == "CCW":
            k = 1 if not unrotate else 3
        elif self.params.rotate_img == "180":
            k = 2
        else:
            raise Exception("Invalid rotate_img option.")
        if type(img) == np.ndarray:
            result = np.rot90(img, k)
        else:
            result = torch.rot90(img, k)
        return result

    def unapply_rotation(self, img):
        return self.apply_rotation(img, unrotate=True)

    def _create_mask(self, img):
        if len(img.shape) == 2:  # image is mono
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # image has alpha channel
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        classes, boxes, scores = self.yolov7_det.detect(img)
        ignore_boxes = []
        keep_boxes = []
        for i, cl in enumerate(classes[0]):
            if self.yolov7_det.names[cl] in self.params.ignore_labels:
                ignore_boxes.append(boxes[0][i])

            if self.yolov7_det.names[cl] in self.params.keep_labels:
                keep_boxes.append(boxes[0][i])

        ignore_mask = np.zeros(img.shape[:2]).astype(np.int8)
        for box in ignore_boxes:
            x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, ignore_mask.shape[1])
            y1 = min(y1, ignore_mask.shape[0])

            ignore_mask[y0:y1, x0:x1] = np.ones((y1 - y0, x1 - x0)).astype(np.int8)

        if self.params.use_keep_labels:
            keep_mask = np.zeros(img.shape[:2]).astype(np.int8)
            for box in keep_boxes:
                x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, keep_mask.shape[1])
                y1 = min(y1, keep_mask.shape[0])
                keep_mask[y0:y1, x0:x1] = np.ones((y1 - y0, x1 - x0)).astype(np.int8)
        else:
            keep_mask = None

        return ignore_mask, keep_mask

    def _process_img(self, image_bgr, ignore_mask=None, keep_mask=None):
        """Process segmentation on image with filtering.

        Args:
            image_bgr: BGR color image.
            ignore_mask: Optional mask of regions to ignore.
            keep_mask: Optional mask of regions to keep.

        Returns:
            Filtered masks as (N, H, W) numpy array.
        """
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        masks = self._run_segmentation(image_rgb)

        if len(masks) == 0:
            return []

        [numMasks, h, w] = masks.shape

        to_delete = []
        for maskId in range(numMasks):
            mask_this_id = masks[maskId, :, :]

            # filter out small masks
            num_pixels = mask_this_id.astype(np.int8).sum()
            if num_pixels < self._min_mask_pixels(image_bgr.shape):
                to_delete.append(maskId)
                continue

            # filter out ignore mask
            if ignore_mask is not None and np.any(
                np.bitwise_and(mask_this_id.astype(np.int8), ignore_mask)
            ):
                to_delete.append(maskId)
                continue

            if (
                keep_mask is not None
                and self.keep_labels_option == "intersect"
                and (
                    np.bitwise_and(mask_this_id.astype(np.int8), keep_mask).sum()
                    < self.params.keep_mask_minimal_intersection
                    * mask_this_id.astype(np.int8).sum()
                )
            ):
                to_delete.append(maskId)
                continue

        masks = np.delete(masks, to_delete, axis=0)

        return masks

    def mask_bounding_box(self, mask):
        # Find the indices of the True values
        true_indices = np.argwhere(mask)

        if len(true_indices) == 0:
            # No True values found, return None or an appropriate response
            return None

        # Calculate the mean of the indices
        mean_coords = np.mean(true_indices, axis=0)

        # Calculate the width and height based on the min and max indices in each dimension
        min_row, min_col = np.min(true_indices, axis=0)
        max_row, max_col = np.max(true_indices, axis=0)
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        # Define a bounding box around the mean coordinates with the calculated width and height
        min_row = int(max(mean_coords[0] - height // 2, 0))
        max_row = int(min(mean_coords[0] + height // 2, mask.shape[0] - 1))
        min_col = int(max(mean_coords[1] - width // 2, 0))
        max_col = int(min(mean_coords[1] + width // 2, mask.shape[1] - 1))

        return (
            min_col,
            min_row,
            max_col,
            max_row,
        )

    def _min_mask_pixels(self, image_shape):
        from_image_fraction = int(
            self.params.min_mask_image_fraction * image_shape[0] * image_shape[1]
        )
        return max(self.params.min_mask_pixels, from_image_fraction)

    def _get_border_occlusion_edge_mask(self, img_shape):
        """Simple border-only occlusion edge mask for point cloud mode.

        Instead of finding where depth pixels start (which requires a dense
        depth image), just marks border pixels based on the configured fraction.
        """
        h, w = img_shape
        max_dim = max(h, w)
        edge_pixels = int(self.params.occlusion_edge_img_frac * max_dim)

        occlusion_edge_mask = np.zeros((h, w), dtype=bool)
        occlusion_edge_mask[:edge_pixels, :] = True
        occlusion_edge_mask[-edge_pixels:, :] = True
        occlusion_edge_mask[:, :edge_pixels] = True
        occlusion_edge_mask[:, -edge_pixels:] = True
        return occlusion_edge_mask

    def _get_occlusion_edge_mask(self, depth_img):
        occlusion_edge_mask = np.zeros_like(depth_img, dtype=bool)

        # Compute edge pixels from fraction of max image dimension
        max_dim = max(depth_img.shape[0], depth_img.shape[1])
        occlusion_edge_pixels = int(self.params.occlusion_edge_img_frac * max_dim)
        occlusion_edge_max_pixels = int(
            self.params.occlusion_edge_max_img_frac * max_dim
        )

        # Valid depth mask: non-zero and finite (handles both d455 zeros and ZED NaNs)
        valid_depth = (depth_img > 0) & np.isfinite(depth_img)

        xmin_valid = np.argmax(valid_depth, axis=1)
        xmax_valid = depth_img.shape[1] - np.argmax(valid_depth[:, ::-1], axis=1) - 1

        for i in range(depth_img.shape[0]):
            left_edge = min(
                xmin_valid[i] + occlusion_edge_pixels, occlusion_edge_max_pixels
            )
            right_edge = max(
                xmax_valid[i] - occlusion_edge_pixels,
                depth_img.shape[1] - occlusion_edge_max_pixels,
            )
            occlusion_edge_mask[i, :left_edge] = True
            occlusion_edge_mask[i, right_edge:] = True

        ymin_valid = np.argmax(valid_depth, axis=0)
        ymax_valid = depth_img.shape[0] - np.argmax(valid_depth[::-1, :], axis=0) - 1

        for j in range(depth_img.shape[1]):
            top_edge = min(
                ymin_valid[j] + occlusion_edge_pixels, occlusion_edge_max_pixels
            )
            bottom_edge = max(
                ymax_valid[j] - occlusion_edge_pixels,
                depth_img.shape[0] - occlusion_edge_max_pixels,
            )
            occlusion_edge_mask[:top_edge, j] = True
            occlusion_edge_mask[bottom_edge:, j] = True

        return occlusion_edge_mask

    def _compute_occlusion_points(
        self, points: np.ndarray, mask: np.ndarray, depth_img, occlusion_edge_mask=None
    ):
        if occlusion_edge_mask is None:
            if self.params.use_point_cloud:
                occlusion_edge_mask = self._get_border_occlusion_edge_mask(
                    mask.shape[:2]
                )
            else:
                occlusion_edge_mask = self._get_occlusion_edge_mask(depth_img)

        occluded_points = points[
            (points[:, 2] < self.params.max_depth)
            & (points[:, 2] > self.params.occlusion_max_depth)
        ]

        # For point cloud mode, skip depth-image-based occlusion pixel extraction
        # since we don't have a dense depth image to index into.
        if occlusion_edge_mask.size > 0:
            occluded_pixels = np.array(
                np.where(np.bitwise_and(mask.astype(bool), occlusion_edge_mask))
            ).T  # (y, x)
            # no edge occlusions for this mask, return just depth-based points
            if occluded_pixels.size == 0:
                return occluded_points

            occluded_pixels_depths = (
                depth_img[occluded_pixels[:, 0], occluded_pixels[:, 1]]
                / self.params.depth_scale
            )

            # Filter out invalid depth values (0, NaN, inf)
            valid_depth_mask = (occluded_pixels_depths > 0) & np.isfinite(
                occluded_pixels_depths
            )
            occluded_pixels = occluded_pixels[valid_depth_mask]
            occluded_pixels_depths = occluded_pixels_depths[valid_depth_mask]

            occluded_pixels_3d_cam = pixel_depth_2_xyz(
                occluded_pixels[:, 1],
                occluded_pixels[:, 0],
                occluded_pixels_depths,
                self.depth_cam_params.K,
            ).T

            # Voxel-downsample to sparsify dense border-based occluded points
            if occluded_pixels_3d_cam.shape[0] > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(occluded_pixels_3d_cam)
                pcd = pcd.voxel_down_sample(voxel_size=self.params.voxel_size)
                occluded_pixels_3d_cam = np.asarray(pcd.points)
            occluded_pixels_3d_cam = occluded_pixels_3d_cam[
                (occluded_pixels_3d_cam[:, 2] < self.params.max_depth)
            ]

            occluded_points = (
                np.vstack([occluded_points, occluded_pixels_3d_cam])
                if occluded_points.shape[0] > 0
                else occluded_pixels_3d_cam
            )
        return occluded_points

    def _remove_point_cloud_outliers(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        if self.params.outlier_removal_std is not None:
            pcd, _ = pcd.remove_statistical_outlier(10, self.params.outlier_removal_std)

        points = np.asarray(pcd.points)

        if self.params.outlier_removal_dbscan_eps is not None and points.size > 0:
            # Perform DBSCAN clustering
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=self.params.outlier_removal_dbscan_eps,
                    min_points=self.params.outlier_removal_dbscan_min_points,
                )
            )

            # Number of clusters, ignoring noise if present
            max_label = labels.max()

            # get largest cluster
            cluster_sizes = np.zeros(max_label + 1)
            for i in range(max_label + 1):
                cluster_sizes[i] = np.sum(labels == i)
            if cluster_sizes.shape[0] == 0:
                return np.array([])
            max_cluster = np.argmax(cluster_sizes)

            # Filter out any points not belonging to max cluster
            filtered_indices = np.where(labels == max_cluster)[0]
            points = points[filtered_indices]
        return np.asarray(points)
