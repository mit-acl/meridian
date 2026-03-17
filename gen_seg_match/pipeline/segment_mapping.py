import numpy as np
import os
import argparse
import shutil
import tqdm
import time
import cv2 as cv
from dataclasses import dataclass, field
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from gen_seg_match.viz.viz_segments import render3d_on_img

from robotdatapy.data.robot_data import NoDataNearTimeException

from gen_seg_match.params import (
    SegmentMappingParams,
    SegmentMappingDataParams,
    SegmenterParams,
)
from gen_seg_match.pipeline.data import SegmentMappingData
from gen_seg_match.map3d.segmenter import Segmenter
from gen_seg_match.map3d.segment_mapper import SegmentMapper
from gen_seg_match.map3d.map import SegmentMap


@dataclass
class SegmentMapping:
    mapping_params: SegmentMappingParams
    segmenter: Segmenter
    mapper: SegmentMapper
    vid_img: bool = False
    vid_3d: bool = False
    show_occluded: bool = False
    output_dir: str = None
    _video_writer: object = field(default=None, init=False, repr=False)

    def run(self, data: SegmentMappingData):
        if data.use_point_cloud:
            t0 = max(
                data.img_data.t0, data.point_cloud_data.t0, data.camera_pose_data.t0
            )
            tf = min(
                data.img_data.tf, data.point_cloud_data.tf, data.camera_pose_data.tf
            )
        else:
            t0 = max(data.img_data.t0, data.depth_data.t0, data.camera_pose_data.t0)
            tf = min(data.img_data.tf, data.depth_data.tf, data.camera_pose_data.tf)

        times = np.arange(t0, tf, self.mapping_params.dt)
        print(f"Processing {len(times)} frames from t={t0:.2f} to t={tf:.2f}")

        for t in tqdm.tqdm(times, desc="Mapping"):
            try:
                img_t = data.img_data.nearest_time(t)
                img = data.img_data.img(img_t)
                pose = data.camera_pose_data.pose(img_t)
                if data.use_point_cloud:
                    pcl = data.align_point_cloud.aligned_point_cloud(img_t)
                    pcl_proj = data.align_point_cloud.projected_point_cloud(pcl)
                    depth = data.align_point_cloud.filter_point_cloud_and_projection(
                        pcl, pcl_proj
                    )
                else:
                    depth = data.depth_data.img(img_t)
            except NoDataNearTimeException:
                continue

            observations, frame_descriptor = self.segmenter.segment(
                img, img_t, pose, depth
            )
            self.mapper.update(img_t, pose, observations, frame_descriptor)

            if self._video_writer is not None:
                frame = self._draw(img_t, img, pose)
                if frame is not None:
                    self._video_writer.write(frame)

    def _draw(self, t, img, pose_cam):
        panes = []
        if self.vid_img:
            panes.append(self._draw_map_on_img(t, pose_cam, img))
        if self.vid_3d:
            panes.append(self._draw_3d(t, pose_cam))
        if not panes:
            return None
        return np.hstack(panes)

    def _draw_map_on_img(self, t, pose_cam, img):
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        viz = img.copy()

        graveyard_time = self.mapping_params.segment_graveyard_time
        all_segs = (
            self.mapper.segments
            + self.mapper.inactive_segments
            + self.mapper.segment_graveyard
        )
        for seg in all_segs:
            if seg.last_seen < t - graveyard_time - 10:
                continue
            outline = seg.outline_2d(pose_cam)
            if outline is None:
                continue
            color = seg.viz_color[::-1]  # RGB → BGR
            for i in range(len(outline) - 1):
                start_point = tuple(outline[i].astype(np.int32))
                end_point = tuple(outline[i + 1].astype(np.int32))
                viz = cv.line(viz, start_point, end_point, color, thickness=2)
            viz = cv.putText(
                viz,
                str(seg.id),
                (np.array(outline[0]) + np.array([10.0, 10.0])).astype(np.int32),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return viz

    def _draw_3d(self, t, pose_cam):
        # Build point clouds from segments in time window
        pcd_list = []
        all_segs = (
            self.mapper.segments
            + self.mapper.inactive_segments
            + self.mapper.segment_graveyard
        )
        for seg in all_segs:
            if seg.last_seen < t - 15.0 or seg.first_seen > t:
                continue
            if seg.points is not None and seg.points.shape[0] > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(seg.points)
                color = np.array(seg.viz_color) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(
                    np.repeat(color.reshape(1, 3), seg.points.shape[0], axis=0)
                )
                pcd_list.append(pcd)

            if self.show_occluded and seg.occluded_points.shape[0] > 0:
                occ_pcd = o3d.geometry.PointCloud()
                occ_pcd.points = o3d.utility.Vector3dVector(seg.occluded_points)
                occ_pcd.colors = o3d.utility.Vector3dVector(
                    np.zeros((seg.occluded_points.shape[0], 3))
                )
                pcd_list.append(occ_pcd)

        # Build pose axes from trajectory
        poses_list = []
        displayed_positions = []
        for i, cam_pose in enumerate(self.mapper.poses_cam_history):
            pose_t = self.mapper.times_history[i]
            if pose_t < t - 15.0 or pose_t > t:
                continue
            if (
                displayed_positions
                and np.linalg.norm(cam_pose[:3, 3] - np.array(displayed_positions[-1]))
                < 0.5
            ):
                continue
            displayed_positions.append(cam_pose[:3, 3])
            pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            pose_obj.transform(cam_pose)
            poses_list.append(pose_obj)

        # Compute behind-camera viewpoint (camera frame: x-right, y-down, z-forward)
        behind_m = 5.0
        above_m = 3.0
        downward_angle = 15.0
        R_cam_behind = Rot.from_euler("x", -downward_angle, degrees=True).as_matrix()
        T_cam_behind = np.eye(4)
        T_cam_behind[:3, :3] = R_cam_behind
        T_cam_behind[:3, 3] = np.array([0.0, -above_m, -behind_m])
        behind_camera_pose = pose_cam @ T_cam_behind

        return render3d_on_img(
            pcd_list + poses_list, self.mapper.camera_params, behind_camera_pose
        )

    def get_segment_map(self) -> SegmentMap:
        return SegmentMap(
            segments=self.mapper.get_segment_map(),
            trajectory=self.mapper.poses_cam_history,
            times=self.mapper.times_history,
            descriptors=self.mapper.frame_descriptors_history
            if self.mapper.frame_descriptors_history
            else None,
        )


def segment_mapping(
    params_path: str,
    output_dir: str,
    run: str = None,
    vid_img: bool = False,
    vid_3d: bool = False,
    show_occluded: bool = False,
):
    print("Loading parameters...")
    mapping_params = SegmentMappingParams.load(params_path, run=run)
    data_params = SegmentMappingDataParams.load(params_path, run=run)
    segmenter_params = SegmenterParams.load(params_path, run=run)
    if data_params.point_cloud_data:
        segmenter_params.use_point_cloud = True
    else:
        segmenter_params.depth_scale = 1 / data_params.depth_scale

    os.makedirs(output_dir, exist_ok=True)

    # Get full time range from bag metadata
    print("Determining bag time range...")
    bag_t_range = SegmentMappingData.get_bag_time_range(data_params)
    if bag_t_range is not None:
        full_t0, full_tf = bag_t_range
        print(f"Bag time range: {full_t0:.2f} to {full_tf:.2f}")
    else:
        full_t0, full_tf = None, None

    # Load a small data slice first to get camera params
    print("Loading initial data to get camera params...")
    if data_params.max_time is not None and full_t0 is not None:
        init_time_range = (
            full_t0,
            full_t0 + min(data_params.max_time, full_tf - full_t0),
        )
    else:
        init_time_range = None

    init_data = SegmentMappingData.from_params(data_params, time_range=init_time_range)
    camera_params = init_data.img_data.camera_params

    # Create segmenter and mapper
    print("Setting up segmenter and mapper...")
    segmenter = Segmenter(segmenter_params, depth_cam_params=camera_params)
    mapper = SegmentMapper(mapping_params, camera_params)
    pipeline = SegmentMapping(
        mapping_params=mapping_params,
        segmenter=segmenter,
        mapper=mapper,
        vid_img=vid_img,
        vid_3d=vid_3d,
        show_occluded=show_occluded,
        output_dir=output_dir,
    )

    # Set up video writer if any video pane is enabled
    num_panes = int(vid_img) + int(vid_3d)
    if num_panes > 0:
        fps = max(5, int(1.0 / mapping_params.dt))
        frame_w = camera_params.width * num_panes
        frame_h = camera_params.height
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        vid_path = os.path.join(output_dir, "segment_mapping.mp4")
        pipeline._video_writer = cv.VideoWriter(
            vid_path, fourcc, fps, (frame_w, frame_h)
        )
        print(f"Recording video to {vid_path} ({frame_w}x{frame_h} @ {fps} fps)")

    wc_t0 = time.time()

    if data_params.max_time is None or full_t0 is None:
        # No chunking — run on initially loaded data
        print("Running mapping (no chunking)...")
        pipeline.run(init_data)
    else:
        # Chunked loading — mapper stays alive between chunks
        chunk_start = full_t0
        chunk_idx = 0

        # Use initial data for the first chunk
        print(
            f"Running mapping chunk {chunk_idx} ({full_t0:.2f} to {init_time_range[1]:.2f})..."
        )
        pipeline.run(init_data)
        del init_data
        chunk_start = init_time_range[1]
        chunk_idx += 1

        while chunk_start < full_tf:
            chunk_end = min(chunk_start + data_params.max_time, full_tf)
            print(
                f"Running mapping chunk {chunk_idx} ({chunk_start:.2f} to {chunk_end:.2f})..."
            )
            data = SegmentMappingData.from_params(
                data_params, time_range=(chunk_start, chunk_end)
            )
            pipeline.run(data)
            del data
            chunk_start = chunk_end
            chunk_idx += 1

    print(f"Mapping took {time.time() - wc_t0:.2f} seconds")

    # Release video writer
    if pipeline._video_writer is not None:
        pipeline._video_writer.release()
        print(f"Saved video to {os.path.join(output_dir, 'segment_mapping.mp4')}")

    # Build and save segment map
    print("Building segment map...")
    segment_map = pipeline.get_segment_map()
    print(
        f"Segment map has {len(segment_map.segments)} segments, "
        f"{len(segment_map.trajectory)} poses"
    )

    map_path = os.path.join(output_dir, "segment_map.pkl")
    segment_map.save(map_path)
    print(f"Saved segment map to {map_path}")

    # Render 3D visualization
    try:
        from gen_seg_match.viz.viz_map import render_segment_map_image

        print("Rendering 3D visualization...")
        viz_img = render_segment_map_image(segment_map)
        viz_path = os.path.join(output_dir, "segment_map_3d.png")
        cv.imwrite(viz_path, viz_img)
        print(f"Saved 3D visualization to {viz_path}")
    except Exception as e:
        print(f"Warning: could not render 3D visualization: {e}")

    # Copy params to output dir
    if os.path.isfile(params_path):
        shutil.copy2(
            params_path, os.path.join(output_dir, os.path.basename(params_path))
        )
    else:
        shutil.copytree(params_path, output_dir, dirs_exist_ok=True)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segment mapping pipeline.")
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
    parser.add_argument(
        "-r",
        "--run",
        type=str,
        default=None,
        help="Run name (for multi-run param files).",
    )
    parser.add_argument(
        "-v",
        "--vid-img",
        action="store_true",
        help="Record video with segment outlines overlaid on image.",
    )
    parser.add_argument(
        "-3",
        "--vid-3d",
        action="store_true",
        help="Record video with 3D point cloud from behind camera.",
    )
    parser.add_argument(
        "--show-occluded",
        action="store_true",
        help="Draw occluded points as black in 3D video pane.",
    )
    args = parser.parse_args()

    segment_mapping(
        args.params,
        args.output,
        run=args.run,
        vid_img=args.vid_img,
        vid_3d=args.vid_3d,
        show_occluded=args.show_occluded,
    )
