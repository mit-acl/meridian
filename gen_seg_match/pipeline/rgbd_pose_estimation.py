import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import robotdatapy as rdp
import time
from robotdatapy.data import PoseData, ImgData
import cv2 as cv
import os
import argparse
import trimesh

from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.params.fastsam_params import FastSAMParams

from gen_seg_match.segment.segment_types import SegmentList
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.map3d.segments_from_img import (
    get_segments_with_occlusion,
    roman_segments_to_general_segments,
)
from gen_seg_match.params import (
    SegmentMatchParams,
    RGBDPoseEstimationParams,
    RomanConversionParams,
    RGBDPoseEstimationDataParams,
)
from gen_seg_match.viz.utils import color_from_seed
from gen_seg_match.viz.img_sparse_viz import img_sparse_viz
from gen_seg_match.pipeline.data import RGBDPoseEstimationData
from gen_seg_match.utils import expandvars_recursive
from gen_seg_match.map3d.segments_from_roman import GeneralSegmentConverter


@dataclass
class RGBDInput:
    time: float
    rgb: np.ndarray
    depth: np.ndarray
    camera_params: rdp.camera.CameraParams
    gravity_direction: np.ndarray = None
    segments: SegmentList = None

    @property
    def bgr(self) -> np.ndarray:
        return self.rgb[:, :, ::-1]

    @property
    def shape(self) -> np.ndarray:
        return self.rgb.shape


@dataclass
class PoseEstimationResult:
    pose_estimate: np.ndarray = None
    pose_gt: np.ndarray = None
    matched_segment_ids: np.ndarray = None
    runtime: float = None

    @property
    def num_matches(self):
        return (
            int(len(self.matched_segment_ids.flatten()) / 2)
            if self.matched_segment_ids is not None
            else 0
        )


@dataclass
class RGBDPoseEstimation:
    pipeline_params: RGBDPoseEstimationParams
    matcher: SegmentMatcher
    roman_conversion_params: RomanConversionParams = None

    def __post_init__(self):
        self.segment_converter = GeneralSegmentConverter(
            params=self.roman_conversion_params
        )

    def rgbd_pose_estimation(self, input1: RGBDInput, input2: RGBDInput):
        t0 = time.time()
        associated_ids = self.matcher.match(
            input1.segments,
            input2.segments,
            input1.gravity_direction,
            input2.gravity_direction,
        )
        tf = time.time()

        return PoseEstimationResult(
            pose_estimate=None,
            pose_gt=None,
            matched_segment_ids=associated_ids,
            runtime=tf - t0,
        )

    def batch_extract_segments(
        self, inputs: List[RGBDInput], segmenter: FastSAMWrapper
    ) -> List[RGBDInput]:
        inputs_with_segments = []
        for rgbd_input in inputs:
            raw_observations = segmenter.run(
                rgbd_input.time, np.eye(4), rgbd_input.bgr, rgbd_input.depth
            )
            roman_segments = get_segments_with_occlusion(
                np.eye(4), raw_observations, rgbd_input.depth, rgbd_input.camera_params
            )
            # TODO: roman conversion params as input below
            general_segments = roman_segments_to_general_segments(
                roman_segments, roman_conversion_params=self.roman_conversion_params
            )
            general_segments = (
                self.segment_converter.segment_with_occlusion_to_general_segments(
                    roman_segments
                )
            )
            general_segments = SegmentList(general_segments)
            inputs_with_segments.append(
                RGBDInput(
                    time=rgbd_input.time,
                    rgb=rgbd_input.rgb,
                    depth=rgbd_input.depth,
                    camera_params=rgbd_input.camera_params,
                    gravity_direction=rgbd_input.gravity_direction,
                    segments=general_segments,
                )
            )

        return inputs_with_segments

    def batch_rgbd_pose_estimation(
        self,
        inputs1: List[RGBDInput],
        inputs2: List[RGBDInput],
        segmenter1: FastSAMWrapper,
        segmenter2: FastSAMWrapper,
        gt1: PoseData = None,
        gt2: PoseData = None,
    ):
        inputs1 = self.batch_extract_segments(inputs1, segmenter1)
        inputs2 = self.batch_extract_segments(inputs2, segmenter2)

        for in1 in inputs1:
            for in2 in inputs2:
                if (
                    gt1 is not None
                    and gt2 is not None
                    and self.fov_iou(in1, in2, gt1, gt2)
                    < self.pipeline_params.min_fov_iou
                ):
                    continue

                result = self.rgbd_pose_estimation(in1, in2)

                if self.pipeline_params.viz_img_matches:
                    self.draw_matches(
                        in1,
                        in2,
                        in1.segments,
                        in2.segments,
                        in1.segments.sublist_from_ids(result.matched_segment_ids[:, 0])
                        if result.num_matches > 0
                        else [],
                        in2.segments.sublist_from_ids(result.matched_segment_ids[:, 1])
                        if result.num_matches > 0
                        else [],
                    )

    def draw_matches(
        self,
        input1: RGBDInput,
        input2: RGBDInput,
        segments1: SegmentList,
        segments2: SegmentList,
        segments1_matches: SegmentList,
        segments2_matches: SegmentList,
        save=True,
    ):
        assert input1.rgb.shape == input2.rgb.shape, (
            "Only inputs of the same shape are currently supported"
        )
        assert len(segments1_matches) == len(segments2_matches)
        output = np.zeros(
            (
                input1.shape[0] * 2 + self.pipeline_params.viz_img_pixel_sep,
                input1.shape[1] * 2 + self.pipeline_params.viz_img_pixel_sep,
                3,
            )
        )

        # draw input segments
        output[: input1.shape[0], : input1.shape[1]] = img_sparse_viz(
            input1.bgr,
            segments1,
            input1.camera_params.K,
            write_ids=self.pipeline_params.viz_write_ids,
        )

        output[
            : input2.shape[0],
            input1.shape[1] + self.pipeline_params.viz_img_pixel_sep :,
        ] = img_sparse_viz(
            input2.bgr,
            segments2,
            input2.camera_params.K,
            write_ids=self.pipeline_params.viz_write_ids,
        )

        colors = [
            color_from_seed(np.random.randint(int(1e9)), order="brg", num_type=int)
            for _ in range(len(segments1_matches))
        ]
        output[
            input1.shape[0] + self.pipeline_params.viz_img_pixel_sep :,
            : input1.shape[1],
        ] = img_sparse_viz(
            input1.bgr,
            segments1_matches,
            input1.camera_params.K,
            colors=colors,
            write_ids=self.pipeline_params.viz_write_ids,
        )

        output[
            input1.shape[0] + self.pipeline_params.viz_img_pixel_sep :,
            input1.shape[1] + self.pipeline_params.viz_img_pixel_sep :,
        ] = img_sparse_viz(
            input2.bgr,
            segments2_matches,
            input2.camera_params.K,
            colors=colors,
            write_ids=self.pipeline_params.viz_write_ids,
        )

        if save:
            file_name = f"{self.pipeline_params.output_directory}/{input1.time}_{input2.time}.png"
            cv.imwrite(file_name, output)

        return output

    def data_to_rgbd_input(self, data: RGBDPoseEstimationData):
        t0s = [data.img_data.t0, data.depth_data.t0]
        tfs = [data.img_data.tf, data.depth_data.tf]
        for pose_data in [data.camera_gt_pose_data, data.camera_est_pose_data]:
            if pose_data is not None:
                t0s.append(pose_data.t0)
                tfs.append(pose_data.tf)
        t0 = np.max(t0s)
        tf = np.min(tfs)

        # get sample image times
        times = []
        img_idx_t0 = data.img_data.idx(t0, force_single=True)
        img_idx_tf = data.img_data.idx(tf, force_single=True)
        if (
            data.camera_est_pose_data is not None
            or data.camera_gt_pose_data is not None
        ):
            pose_data_ref = (
                data.camera_gt_pose_data
                if data.camera_gt_pose_data is not None
                else data.camera_est_pose_data
            )
            last_position = None
            img_idx = img_idx_t0
            while img_idx < img_idx_tf:
                curr_position = pose_data_ref.position(data.img_data.times[img_idx])
                if (
                    last_position is None
                    or np.linalg.norm(curr_position - last_position)
                    > self.pipeline_params.sample_distance
                ):
                    last_position = curr_position
                    times.append(data.img_data.times[img_idx])
                img_idx += 1
        else:
            times = data.img_data.times[img_idx_t0:img_idx_tf]

        rgbd_inputs = []
        for t in times:
            gravity_direction = data.gravity_direction
            if gravity_direction is not None:
                gravity_world = gravity_direction
                if data.camera_est_pose_data is not None:
                    T_world_cam = data.camera_est_pose_data.pose(t)
                else:
                    T_world_cam = data.camera_gt_pose_data.pose(t)
                gravity_cam = T_world_cam[:3, :3].T @ gravity_world
                gravity_direction = gravity_cam
            rgbd_inputs.append(
                RGBDInput(
                    time=t,
                    rgb=data.img_data.img(t)[:, :, ::-1],
                    depth=data.depth_data.img(t),
                    camera_params=data.img_data.camera_params,
                    gravity_direction=gravity_direction,
                )
            )
        return rgbd_inputs

    def fov_iou(
        self,
        input1: RGBDInput,
        input2: RGBDInput,
        gt1: PoseData,
        gt2: PoseData,
    ) -> float:
        """
        Returns the fraction of the field of views that overlap between two RGBD inputs.
        Specifically, this is the volume of the intersection of the two FOVs divided by
        the volume of the union of the two FOVs. Max depth is used to limit the FOVs.

        Args:
            input1 (RGBDInput): RGBD image input 1.
            input2 (RGBDInput): RGBD image input 2.
            gt1 (PoseData): Ground truth pose data for input 1.
            gt2 (PoseData): Ground truth pose data for input 2.

        Returns:
            float: Field of view intersection over union.
        """
        # get camera poses
        T_world_cam1 = gt1.pose(input1.time)
        T_world_cam2 = gt2.pose(input2.time)

        # get frustums
        frustum1 = self.get_camera_frustum(input1, T_world_cam1)
        frustum2 = self.get_camera_frustum(input2, T_world_cam2)

        intersection_mesh = frustum1.intersection(frustum2)
        if not intersection_mesh:
            return 0.0

        intersection_volume = intersection_mesh.volume
        union_volume = frustum1.volume + frustum2.volume - intersection_volume

        return intersection_volume / union_volume

    def get_camera_frustum(
        self, rgbd_input: RGBDInput, gt_pose: np.ndarray
    ) -> np.ndarray:
        """
        Returns the 3D points representing the camera frustum for the given RGBD input
        and ground truth pose.

        Args:
            rgbd_input (RGBDInput): RGBD image input.
            gt_pose (np.ndarray): Ground truth camera pose as a 4x4 transformation matrix.
        Returns:
            trimesh.Trimesh: 3D mesh shape representing the camera frustum.
        """
        base_pixels = np.array(
            [
                [0, 0],
                [rgbd_input.shape[1], 0],
                [rgbd_input.shape[1], rgbd_input.shape[0]],
                [0, rgbd_input.shape[0]],
            ]
        )
        base_points_cam = rdp.camera.pixel_depth_2_xyz(
            base_pixels[:, 0],
            base_pixels[:, 1],
            depth=np.repeat([self.pipeline_params.max_fov_depth], 4),
            K=rgbd_input.camera_params.K,
        ).T  # shape = (4, 3)
        pyramid_points_cam = np.vstack(
            [
                base_points_cam,
                np.array([[0, 0, 0]]),  # camera center
            ]
        )  # shape = (5, 3)
        pyramid_points_world = rdp.transform.transform(gt_pose, pyramid_points_cam)
        frustum_mesh = trimesh.convex.convex_hull(pyramid_points_world)
        return frustum_mesh


def rgbd_pose_estimation(params, output_dir, runs=Tuple[str, str]):
    pipeline_params = RGBDPoseEstimationParams.load(params)
    pipeline_params.output_directory = output_dir
    runner = RGBDPoseEstimation(
        pipeline_params=pipeline_params,
        matcher=SegmentMatcher(SegmentMatchParams.load(params)),
        roman_conversion_params=RomanConversionParams.load(params),
    )

    rgbd_input_lists = []
    rgbd_data = []
    segmenters = []

    for run in runs:
        os.environ["RUN"] = run
        data_params = RGBDPoseEstimationDataParams.load(params, run=run)
        rgbd_data.append(RGBDPoseEstimationData.from_params(data_params))
        rgbd_input_lists.append(runner.data_to_rgbd_input(rgbd_data[-1]))
        segmenters.append(
            FastSAMWrapper.from_params(
                FastSAMParams(
                    semantics="dino",
                    device="cuda",
                    max_depth=8.0,
                    plane_filter_params=tuple([np.inf, 1.0, 0.2]),
                    conf=0.2,
                    iou=0.5,
                    max_mask_len_div=1,
                    erosion_size=3,
                ),
                rgbd_data[-1].depth_data.camera_params,
            )
        )

    runner.batch_rgbd_pose_estimation(
        rgbd_input_lists[0],
        rgbd_input_lists[1],
        segmenters[0],
        segmenters[1],
        rgbd_data[0].camera_gt_pose_data,
        rgbd_data[1].camera_gt_pose_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "-r", "--runs", type=str, nargs=2, required=True, help="Run names."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(expandvars_recursive(args.output))

    rgbd_pose_estimation(args.params, args.output, args.runs)
