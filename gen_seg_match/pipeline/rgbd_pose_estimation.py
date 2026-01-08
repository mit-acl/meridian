import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Any
import robotdatapy as rdp
import time
from robotdatapy.data import PoseData, ImgData
import cv2 as cv
import os
import argparse
import trimesh
import pickle
import tqdm
import shutil

from gen_seg_match.segment.segment_types import SegmentList, DenseSegment
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.map3d.segments_from_img import (
    get_segments_with_occlusion,
    roman_segments_to_general_segments,
)
from gen_seg_match.map3d.segmenter import Segmenter
from gen_seg_match.params import (
    SegmentMatchParams,
    RGBDPoseEstimationParams,
    RomanConversionParams,
    RGBDPoseEstimationDataParams,
    SegmenterParams,
    RegisterParams,
)
from gen_seg_match.viz.utils import color_from_seed
from gen_seg_match.viz.img_sparse_viz import img_sparse_viz
from gen_seg_match.viz.viz_segments import viz_masks_on_img
from gen_seg_match.pipeline.data import RGBDPoseEstimationData
from gen_seg_match.pipeline.result import (
    PoseEstimationResult,
    PoseEstimationResultMatrix,
)
from gen_seg_match.utils import expandvars_recursive
from gen_seg_match.map3d.segments_from_roman import GeneralSegmentConverter
from gen_seg_match.register.registerer import (
    Registerer,
    InsufficientAssociationsException,
)


@dataclass
class RGBDInput:
    time: float
    rgb: np.ndarray
    depth: np.ndarray
    camera_params: rdp.camera.CameraParams
    gravity_direction: np.ndarray = None
    segments: SegmentList = None
    pose_gt: np.ndarray = None  # optional ground truth pose

    @property
    def bgr(self) -> np.ndarray:
        return self.rgb[:, :, ::-1]

    @property
    def shape(self) -> np.ndarray:
        return self.rgb.shape


@dataclass
class RGBDPoseEstimation:
    pipeline_params: RGBDPoseEstimationParams
    matcher: SegmentMatcher
    registerer: Registerer
    roman_conversion_params: RomanConversionParams = None

    def __post_init__(self):
        self.segment_converter = GeneralSegmentConverter(
            params=self.roman_conversion_params
        )
        for dir_path in [
            self.pipeline_params.output_directory,
            self.segment_directory,
            f"{self.segment_directory}/run1",
            f"{self.segment_directory}/run2",
            self.match_directory,
        ]:
            if not os.path.isdir(expandvars_recursive(dir_path)):
                os.mkdir(expandvars_recursive(dir_path))

    @property
    def segment_directory(self):
        return os.path.join(self.pipeline_params.output_directory, "segment")

    @property
    def match_directory(self):
        return os.path.join(self.pipeline_params.output_directory, "match")

    def rgbd_pose_estimation(self, input1: RGBDInput, input2: RGBDInput):
        t0 = time.time()
        associated_ids = self.matcher.match(
            input1.segments,
            input2.segments,
            input1.gravity_direction,
            input2.gravity_direction,
        )
        try:
            transformation = self.registerer.register(
                input1.segments,
                input2.segments,
                input1.gravity_direction,
                input2.gravity_direction,
                correspondences=associated_ids,
            ).transformation
        except InsufficientAssociationsException:
            transformation = np.zeros((4, 4)) * np.nan

        tf = time.time()

        T_i_j = None
        if input1.pose_gt is not None and input2.pose_gt is not None:
            T_i_j = np.linalg.inv(input1.pose_gt) @ input2.pose_gt
        return PoseEstimationResult(
            T_i_j=T_i_j,
            T_i_j_hat=transformation,
            associations=associated_ids,
            runtime_s=tf - t0,
        )

    def batch_extract_segments(
        self, inputs: List[RGBDInput], segmenter: Segmenter, output_dir: str = None
    ) -> List[RGBDInput]:
        for i, rgbd_input in enumerate(inputs):
            raw_observations, _ = segmenter.segment(
                rgbd_input.bgr, rgbd_input.time, np.eye(4), rgbd_input.depth
            )
            dense_segments = [
                DenseSegment.from_observation(obs) for obs in raw_observations
            ]
            general_segments = (
                self.segment_converter.segment_with_occlusion_to_general_segments(
                    dense_segments
                )
            )
            general_segments = SegmentList(general_segments)
            rgbd_input.segments = general_segments
            if output_dir is not None:
                self.draw_segments(
                    rgbd_input,
                    raw_observations,
                    general_segments,
                    output_file=f"{output_dir}/{i}.png",
                )

        if output_dir is not None:
            with open(f"{output_dir}/segments.pkl", "wb") as f:
                pickle.dump(rgbd_input, f)

        return inputs

    def batch_rgbd_pose_estimation(
        self,
        inputs1: List[RGBDInput],
        inputs2: List[RGBDInput],
        segmenter: Segmenter = None,
        depth_camera_params1: rdp.camera.CameraParams = None,
        depth_camera_params2: rdp.camera.CameraParams = None,
        has_segments: bool = False,
    ):
        assert (
            None not in [segmenter, depth_camera_params1, depth_camera_params2]
        ) or has_segments, (
            "Either segmenter must be provided or inputs must already have segments."
        )

        if not has_segments:
            segmenter.set_depth_camera_params(depth_camera_params1)
            inputs1 = self.batch_extract_segments(
                inputs1, segmenter, output_dir=f"{self.segment_directory}/run1"
            )
            segmenter.set_depth_camera_params(depth_camera_params2)
            inputs2 = self.batch_extract_segments(
                inputs2, segmenter, output_dir=f"{self.segment_directory}/run2"
            )

        results_matrix = PoseEstimationResultMatrix((len(inputs1), len(inputs2)))

        for i, in1 in enumerate(tqdm.tqdm(inputs1)):
            for j, in2 in enumerate(inputs2):
                if (
                    in1.pose_gt is not None
                    and in2.pose_gt is not None
                    and self.fov_iou(in1, in2) < self.pipeline_params.min_fov_iou
                ):
                    continue

                result = self.rgbd_pose_estimation(in1, in2)

                if self.pipeline_params.viz_img_matches:
                    output_file = f"{self.match_directory}/{i}_{j}.png"
                    self.draw_matches(
                        in1,
                        in2,
                        in1.segments,
                        in2.segments,
                        in1.segments.sublist_from_ids(result.associations[:, 0])
                        if result.num_associations > 0
                        else [],
                        in2.segments.sublist_from_ids(result.associations[:, 1])
                        if result.num_associations > 0
                        else [],
                        output_file=output_file,
                    )

                results_matrix[i, j] = result

        results_matrix.save(f"{self.match_directory}/results.npz")
        results_matrix.plot()
        plt.savefig(f"{self.match_directory}/results.png")
        plt.close()
        # results_matrix.plot_point_vs_line_associations()
        # plt.savefig(run_output_dir / "point_vs_line_associations.png")
        # plt.close()

        return inputs1, inputs2

    def draw_matches(
        self,
        input1: RGBDInput,
        input2: RGBDInput,
        segments1: SegmentList,
        segments2: SegmentList,
        segments1_matches: SegmentList,
        segments2_matches: SegmentList,
        output_file: str = None,
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

        if output_file is not None:
            cv.imwrite(output_file, output)

        return output

    def draw_segments(
        self,
        rgbd_input: RGBDInput,
        observations,
        segments: SegmentList,
        output_file: str = None,
    ):
        output = np.zeros(
            (
                rgbd_input.shape[0],
                rgbd_input.shape[1] * 2 + self.pipeline_params.viz_img_pixel_sep,
                3,
            )
        )

        # draw raw observations
        output[:, : rgbd_input.shape[1]] = viz_masks_on_img(
            rgbd_input.bgr, observations, alpha=0.5
        )

        # draw segments
        output[
            :,
            rgbd_input.shape[1] + self.pipeline_params.viz_img_pixel_sep :,
        ] = img_sparse_viz(
            rgbd_input.bgr,
            segments,
            rgbd_input.camera_params.K,
            write_ids=self.pipeline_params.viz_write_ids,
        )

        if output_file is not None:
            cv.imwrite(output_file, output)

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
            pose_gt = (
                data.camera_gt_pose_data.pose(t)
                if data.camera_gt_pose_data is not None
                else None
            )
            rgbd_inputs.append(
                RGBDInput(
                    time=t,
                    rgb=data.img_data.img(t)[:, :, ::-1],
                    depth=data.depth_data.img(t),
                    camera_params=data.img_data.camera_params,
                    gravity_direction=gravity_direction,
                    pose_gt=pose_gt,
                )
            )
        return rgbd_inputs

    def fov_iou(
        self,
        input1: RGBDInput,
        input2: RGBDInput,
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
        T_world_cam1 = input1.pose_gt
        T_world_cam2 = input2.pose_gt

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
    ) -> trimesh.Trimesh:
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


def rgbd_pose_estimation(
    params, output_dir, runs=Tuple[str, str], segmentation_dir=None
):
    pipeline_params = RGBDPoseEstimationParams.load(params)
    pipeline_params.output_directory = output_dir
    runner = RGBDPoseEstimation(
        pipeline_params=pipeline_params,
        matcher=SegmentMatcher(SegmentMatchParams.load(params)),
        registerer=Registerer(RegisterParams.load(params)),
        roman_conversion_params=RomanConversionParams.load(params),
    )

    # copy params to output dir
    if os.path.isfile(params):
        shutil.copy2(params, os.path.join(output_dir, os.path.basename(params)))
    else:
        shutil.copytree(params, output_dir, dirs_exist_ok=True)

    rgbd_input_lists = []
    rgbd_data = []
    segmenters = []

    if segmentation_dir is not None:
        for i in range(2):
            with open(f"{segmentation_dir}/run{i + 1}/segments.pkl", "rb") as f:
                rgbd_input_lists.append(pickle.load(f))

        runner.batch_rgbd_pose_estimation(
            rgbd_input_lists[0],
            rgbd_input_lists[1],
            has_segments=True,
        )

    else:
        for run in runs:
            os.environ["RUN"] = run
            data_params = RGBDPoseEstimationDataParams.load(params, run=run)
            rgbd_data.append(RGBDPoseEstimationData.from_params(data_params))
            rgbd_input_lists.append(runner.data_to_rgbd_input(rgbd_data[-1]))
        segmenter = Segmenter(SegmenterParams.load(params))

        runner.batch_rgbd_pose_estimation(
            rgbd_input_lists[0],
            rgbd_input_lists[1],
            segmenter,
            rgbd_data[0].depth_data.camera_params,
            rgbd_data[1].depth_data.camera_params,
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
    parser.add_argument(
        "-s",
        "--segment-dir",
        type=str,
        default=None,
        help="Directory to load precomputed segments.",
    )
    args = parser.parse_args()

    rgbd_pose_estimation(
        args.params, args.output, args.runs, segmentation_dir=args.segment_dir
    )
