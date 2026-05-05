import numpy as np
import os
import argparse
import tqdm
import time
import cv2 as cv
from dataclasses import dataclass, field
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from meridian.viz.viz_segments import render3d_on_img

from robotdatapy.data.robot_data import NoDataNearTimeException

from meridian.params import (
    SegmentMappingParams,
    SegmentMappingDataParams,
    SegmenterParams,
)
from meridian.pipeline.data import SegmentMappingData
from meridian.map3d.segmenter import Segmenter
from meridian.map3d.segment_mapper import SegmentMapper
from meridian.map3d.map import SegmentMap


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
    _timing: dict = field(
        default_factory=lambda: {"data": [], "segment": [], "map": [], "submap_2d": []},
        init=False,
        repr=False,
    )
    _last_descriptor_position: object = field(default=None, init=False, repr=False)

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
            t_data_start = time.time()
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

            # Determine whether to compute frame descriptor based on distance
            position = pose[:3, 3]
            should_compute_descriptor = (
                self._last_descriptor_position is None
                or np.linalg.norm(position - self._last_descriptor_position)
                >= self.mapping_params.frame_descriptor_dist_m
            )

            t_seg_start = time.time()
            observations, frame_descriptor = self.segmenter.segment(
                img,
                img_t,
                pose,
                depth,
                compute_frame_descriptor=should_compute_descriptor,
            )
            if frame_descriptor is not None:
                self._last_descriptor_position = position.copy()

            t_map_start = time.time()
            self.mapper.update(img_t, pose, observations, frame_descriptor)

            t_submap_start = time.time()
            if self.mapping_params.inc_submaps_2d:
                self.mapper.process_submaps_2d(img_t, pose)
            t_end = time.time()

            self._timing["data"].append(t_seg_start - t_data_start)
            self._timing["segment"].append(t_map_start - t_seg_start)
            self._timing["map"].append(t_submap_start - t_map_start)
            self._timing["submap_2d"].append(t_end - t_submap_start)

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
        descriptors = self.mapper.frame_descriptors_history
        # If no descriptors were ever computed, pass None for backward compat
        if not any(d is not None for d in descriptors):
            descriptors = None
        return SegmentMap(
            segments=self.mapper.get_segment_map(),
            trajectory=self.mapper.poses_cam_history,
            times=self.mapper.times_history,
            descriptors=descriptors,
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

    # Set up incremental 2D ground submap pipeline if enabled
    ground_submap_mapping = None
    place_recognition = None
    if mapping_params.inc_submaps_2d:
        from meridian.params import (
            GroundSubmapParams,
            SegmentToPrimitiveConversionParams,
            GroundSegmenterParams,
        )
        from meridian.map2d.ground_segmenter import GroundSegmenter
        from meridian.map2d.segment_to_primitive import SegmentToPrimitiveConverter
        from meridian.map2d.ground_submap_primitive_mapping import (
            GroundSubmapPrimitiveMapping,
        )

        ground_submap_params = GroundSubmapParams.load(params_path, run=run)
        conversion_params = SegmentToPrimitiveConversionParams.load(
            params_path, run=run
        )
        ground_segmenter_params = GroundSegmenterParams.load(params_path, run=run)

        converter = SegmentToPrimitiveConverter(conversion_params)
        ground_segmenter = GroundSegmenter(ground_segmenter_params)

        # Place recognition is optional
        try:
            from meridian.params import CrossViewPlaceRecognitionParams
            from meridian.cross_view.place_recognition import (
                CrossViewPlaceRecognition,
            )

            pr_params = CrossViewPlaceRecognitionParams.load(params_path, run=run)
            place_recognition = CrossViewPlaceRecognition(pr_params)
        except Exception:
            place_recognition = None

        ground_submap_mapping = GroundSubmapPrimitiveMapping(
            ground_submap_params,
            converter,
            ground_segmenter,
            place_recognition,
        )
        print(
            f"Incremental 2D submaps enabled: "
            f"{mapping_params.sm2d_num_segments} segs/submap, "
            f"{mapping_params.sm2d_num_new_segments} new segs trigger"
        )

    mapper = SegmentMapper(
        mapping_params,
        camera_params,
        ground_submap_mapping=ground_submap_mapping,
        place_recognition=place_recognition,
    )
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

    wall_time = time.time() - wc_t0
    print(f"Mapping took {wall_time:.2f} seconds")

    # Save per-frame timing breakdown
    timing = pipeline._timing
    n_frames = len(timing["data"])
    if n_frames > 0:
        mean_data = np.mean(timing["data"])
        mean_seg = np.mean(timing["segment"])
        mean_map = np.mean(timing["map"])
        mean_sm2d = np.mean(timing["submap_2d"])
        mean_total = mean_data + mean_seg + mean_map + mean_sm2d
        timing_path = os.path.join(output_dir, "timing.txt")
        with open(timing_path, "w") as f:
            f.write(f"Frames:           {n_frames}\n")
            f.write(f"Wall-clock time:  {wall_time:.2f}s\n")
            f.write(f"\nPer-frame averages:\n")
            f.write(
                f"  Data fetch:     {mean_data:.4f}s  ({mean_data / mean_total * 100:.1f}%)\n"
            )
            f.write(
                f"  Segmenter:      {mean_seg:.4f}s  ({mean_seg / mean_total * 100:.1f}%)\n"
            )
            f.write(
                f"  Mapper update:  {mean_map:.4f}s  ({mean_map / mean_total * 100:.1f}%)\n"
            )
            if mapping_params.inc_submaps_2d:
                f.write(
                    f"  Submap 2D:      {mean_sm2d:.4f}s  ({mean_sm2d / mean_total * 100:.1f}%)\n"
                )
            f.write(
                f"  Total:          {mean_total:.4f}s  ({1 / mean_total:.1f} fps)\n"
            )
            if mapping_params.inc_submaps_2d:
                f.write(f"\n2D Submaps created: {len(mapper.submaps_2d)}\n")
        print(f"Saved timing breakdown to {timing_path}")

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

    # Save incremental 2D ground submaps
    if mapping_params.inc_submaps_2d and mapper.submaps_2d:
        import pickle

        ground_dir = os.path.join(output_dir, "ground", "segments")
        os.makedirs(ground_dir, exist_ok=True)
        for k, submap_2d in enumerate(mapper.submaps_2d):
            submap_path = os.path.join(ground_dir, f"{k}.pkl")
            submap_2d.save(submap_path)

            # Save dense points if intermediates available
            if k < len(mapper._submap_intermediates):
                intermediate = mapper._submap_intermediates[k]
                if intermediate is not None and intermediate.aerial_segments:
                    dense_dir = os.path.join(ground_dir, f"{k}_dense")
                    os.makedirs(dense_dir, exist_ok=True)
                    ground_submap_params = ground_submap_mapping.submap_params
                    for aerial_seg in intermediate.aerial_segments:
                        dense_path = os.path.join(dense_dir, f"{aerial_seg.id}.pkl")
                        pts = aerial_seg.points
                        max_n = ground_submap_params.dense_points_max_n
                        if max_n is not None and len(pts) > max_n:
                            idx = np.round(np.linspace(0, len(pts) - 1, max_n)).astype(
                                int
                            )
                            pts = pts[idx]
                        with open(dense_path, "wb") as f:
                            pickle.dump(pts, f)

        print(f"Saved {len(mapper.submaps_2d)} ground submaps to {ground_dir}")

        # Save ground submap visualizations
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from meridian.viz.cross_view_viz import viz_ground_segments

        viz_dir = os.path.join(output_dir, "ground", "viz")
        os.makedirs(viz_dir, exist_ok=True)
        for k, submap_2d in enumerate(mapper.submaps_2d):
            if k >= len(mapper._submap_intermediates):
                continue
            intermediate = mapper._submap_intermediates[k]
            if intermediate is None:
                continue
            fig, ax = viz_ground_segments(
                intermediate.flattened_submap,
                intermediate.aerial_segments,
                intermediate.general_segments,
                submap_2d.segments,
                conversion_params.alpha_shape_alpha,
                conversion_params.alpha_shape_grid_downsample,
                conversion_params.alpha_shape_max_n_pts,
                conversion_params.alpha_shape_ref_size_m,
            )
            fig.savefig(os.path.join(viz_dir, f"{k}.png"), dpi=400)
            plt.close(fig)
        print(f"Saved ground submap visualizations to {viz_dir}")

    # Render 3D visualization
    # try:
    #     from meridian.viz.viz_map import render_segment_map_image

    #     print("Rendering 3D visualization...")
    #     viz_img = render_segment_map_image(segment_map)
    #     viz_path = os.path.join(output_dir, "segment_map_3d.png")
    #     cv.imwrite(viz_path, viz_img)
    #     print(f"Saved 3D visualization to {viz_path}")
    # except Exception as e:
    #     print(f"Warning: could not render 3D visualization: {e}")

    # Save all params (including defaults) and commit hash
    from meridian.utils import save_params, save_commit_hash

    save_params(output_dir, mapping_params, data_params, segmenter_params)
    save_commit_hash(output_dir)

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
