import numpy as np
import os
import argparse
import shutil
import tqdm
import time
import cv2 as cv
from dataclasses import dataclass

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

    def run(self, data: SegmentMappingData):
        t0 = max(data.img_data.t0, data.depth_data.t0, data.camera_pose_data.t0)
        tf = min(data.img_data.tf, data.depth_data.tf, data.camera_pose_data.tf)

        times = np.arange(t0, tf, self.mapping_params.dt)
        print(f"Processing {len(times)} frames from t={t0:.2f} to t={tf:.2f}")

        for t in tqdm.tqdm(times, desc="Mapping"):
            try:
                img_t = data.img_data.nearest_time(t)
                img = data.img_data.img(img_t)
                depth = data.depth_data.img(img_t)
                pose = data.camera_pose_data.pose(img_t)
            except NoDataNearTimeException:
                continue

            observations, frame_descriptor = self.segmenter.segment(
                img, img_t, pose, depth
            )
            self.mapper.update(img_t, pose, observations, frame_descriptor)

    def get_segment_map(self) -> SegmentMap:
        return SegmentMap(
            segments=self.mapper.get_segment_map(),
            trajectory=self.mapper.poses_flu_history,
            times=self.mapper.times_history,
            descriptors=self.mapper.frame_descriptors_history
            if self.mapper.frame_descriptors_history
            else None,
        )


def segment_mapping(params_path: str, output_dir: str, run: str = None):
    print("Loading parameters...")
    mapping_params = SegmentMappingParams.load(params_path, run=run)
    data_params = SegmentMappingDataParams.load(params_path, run=run)
    segmenter_params = SegmenterParams.load(params_path, run=run)

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
    mapper.set_T_camera_flu(mapping_params.T_camera_flu)

    pipeline = SegmentMapping(
        mapping_params=mapping_params,
        segmenter=segmenter,
        mapper=mapper,
    )

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
    args = parser.parse_args()

    segment_mapping(args.params, args.output, run=args.run)
