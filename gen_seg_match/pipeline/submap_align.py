import numpy as np
import matplotlib.pyplot as plt
import robotdatapy as rdp
from robotdatapy.data import PoseData
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass, field
from copy import deepcopy
import time
from scipy.spatial.transform import Rotation as Rot
import argparse
from pathlib import Path
import json
from enum import Enum
import pickle

from roman.align.results import SubmapAlignResults, save_submap_align_results
from roman.align.dist_reg_with_pruning import GravityConstraintError
from roman.map.map import ROMANMap
from roman.params.submap_align_params import SubmapAlignInputOutput, SubmapAlignParams

from gen_seg_match.map3d.submap import Submap
from gen_seg_match.match.segment_matcher import (
    SegmentMatcher,
)
from gen_seg_match.params import (
    SubmapParams,
    DenseToSparseParams,
    SegmentMatchParams,
    RegisterParams,
)
from gen_seg_match.map3d.submap import submaps_from_roman_map
from gen_seg_match.map3d.dense_to_sparse_converter import DenseToSparseConverter
from gen_seg_match.utils import expandvars_recursive
from gen_seg_match.segment.segment_types import SegmentList, SegmentLine, SegmentPoint
from gen_seg_match.register.registerer import (
    Registerer,
    InsufficientAssociationsException,
)
from gen_seg_match.pipeline.result import (
    PoseEstimationResult,
    PoseEstimationResultMatrix,
    AssociationType,
)


GRAVITY_DIR_NEG_Z: np.ndarray = np.array([0.0, 0.0, -1.0])


class AssociationType(Enum):
    POINT_TO_POINT = 1
    LINE_TO_LINE = 2


@dataclass
class ROMANBasedSubmapAlignParams:
    max_distance: float = 20.0
    skip_distant_submaps: bool = True
    is_single_robot: bool = False


# TODO: probably want to move this to gsm_tools while we are testing on many different registration methods
def register_submaps(
    submap_1: Submap,
    submap_2: Submap,
    matcher: SegmentMatcher,
    registerer: Registerer,
    T_sm1_sm2_gt: np.ndarray,
    params: ROMANBasedSubmapAlignParams,
):
    result = PoseEstimationResult(T_i_j=T_sm1_sm2_gt)

    # if submaps are close enough (based on gt) run pose estimation
    if result.gt_distance_m < params.max_distance and params.skip_distant_submaps:
        return result

    # compute assocations
    start_t = time.time()

    try:
        associations = matcher.match(
            submap_1.segments,
            submap_2.segments,
            global_z_dir1=GRAVITY_DIR_NEG_Z,
            global_z_dir2=GRAVITY_DIR_NEG_Z,
        )
        association_types = []
        # track association types
        for assoc in associations:
            if type(submap_1.segments.get_segment_from_id(assoc[0])) is SegmentPoint:
                association_types.append(AssociationType.POINT_TO_POINT)
            else:
                association_types.append(AssociationType.LINE_TO_LINE)
        result.associations = associations.copy()
        result.association_types = tuple(association_types)

        T_sm1grav_sm2grav_hat = registerer.register(
            submap_1.segments,
            submap_2.segments,
            GRAVITY_DIR_NEG_Z,
            GRAVITY_DIR_NEG_Z,
            correspondences=associations,
        ).transformation
        # (T^odom_flu)^{-1} @ T^odom_gravaligned
        T_sm1_sm1grav = np.linalg.inv(submap_1.pose_flu) @ submap_1.pose_gravity_aligned
        T_sm2_sm2grav = np.linalg.inv(submap_2.pose_flu) @ submap_2.pose_gravity_aligned
        T_sm1_sm2_hat = (
            T_sm1_sm1grav @ T_sm1grav_sm2grav_hat @ np.linalg.inv(T_sm2_sm2grav)
        )
        result.inlier_ratio = (
            (
                len(associations)
                / np.mean([len(submap_1.segments), len(submap_2.segments)])
            )
            if len(associations) > 0
            else 0.0
        )
        result.T_i_j_hat = T_sm1_sm2_hat

    except (InsufficientAssociationsException, GravityConstraintError):
        result.angle_error_rad = np.inf
        result.translation_error_m = np.inf

    result.runtime_s = time.time() - start_t
    return result


def submap_align(
    submaps_1: List[Submap],
    submaps_2: List[Submap],
    gt_pose_1: PoseData,
    gt_pose_2: PoseData,
    matcher: SegmentMatcher,
    registerer: Registerer,
    params: ROMANBasedSubmapAlignParams = ROMANBasedSubmapAlignParams(),
) -> PoseEstimationResultMatrix:
    results_matrix = PoseEstimationResultMatrix((len(submaps_1), len(submaps_2)))

    for i in tqdm(range(len(submaps_1))):
        for j in range(len(submaps_2)):
            sm_i = deepcopy(submaps_1[i])
            sm_j = deepcopy(submaps_2[j])

            if params.is_single_robot:
                common_ids = set(sm_i.segment_ids).intersection(sm_j.segment_ids)
                for sm in [sm_i, sm_j]:
                    to_rm = [seg for seg in sm.segments if seg.id in common_ids]
                    for seg in to_rm:
                        sm.segments.remove(seg)

            T_w_smi = gt_pose_1.pose(sm_i.time)
            T_w_smj = gt_pose_2.pose(sm_j.time)

            T_smi_smj = np.linalg.inv(T_w_smi) @ T_w_smj
            results_matrix[i, j] = [
                register_submaps(sm_i, sm_j, matcher, registerer, T_smi_smj, params)
            ]

    return results_matrix


def batch_submap_align(
    run_names: List[str],
    submap_lists: List[List[Submap]],
    gt_poses: List[PoseData],
    matcher: SegmentMatcher,
    registerer: Registerer,
    submap_align_params: ROMANBasedSubmapAlignParams = ROMANBasedSubmapAlignParams(),
    output_dir: Path = None,
    roman_maps: List[ROMANMap] = None,
) -> Dict[str, Dict[str, SubmapAlignResults]]:
    results_dict = {name_i: {} for name_i in run_names}
    for i in range(len(run_names)):
        for j in range(i, len(run_names)):
            if i == j:
                continue
            results_ij = submap_align(
                submap_lists[i],
                submap_lists[j],
                gt_poses[i],
                gt_poses[j],
                matcher,
                registerer,
                submap_align_params,
            )
            if output_dir is not None and roman_maps is not None:
                run_output_dir = output_dir / f"{run_names[i]}_{run_names[j]}"
                run_output_dir.mkdir(parents=True, exist_ok=True)
                results_ij.save(str(run_output_dir / "results.npz"))
                results_ij.plot()
                plt.savefig(run_output_dir / "results.png")
                plt.close()
                results_ij.plot_point_vs_line_associations()
                plt.savefig(run_output_dir / "point_vs_line_associations.png")
                plt.close()
            results_dict[run_names[i]][run_names[j]] = results_ij
    return results_dict


if __name__ == "__main__":
    import yaml
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--roman-results-dir",
        type=str,
        required=True,
        help="ROMAN results directory. All files in map/*.pkl will be loaded as maps, "
        + "unless specific run names are given.",
    )
    parser.add_argument("-p", "--params", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("-n", "--run-names", type=str, nargs="+", default=None)
    parser.add_argument("-e", "--run-env", type=str, default="RUN")
    parser.add_argument("--save-general-segments", action="store_true")
    args = parser.parse_args()

    output_dir = Path(expandvars_recursive(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    # copy params to output dir
    os.system(f"cp {expandvars_recursive(args.params)} {output_dir / 'params.yaml'}")
    with open(expandvars_recursive(args.params), "r") as f:
        params_dict = yaml.full_load(f)

    # load ROMAN maps
    roman_results_dir = Path(expandvars_recursive(args.roman_results_dir))
    map_dir = roman_results_dir / "map"
    if args.run_names is None:
        if "run_names" in params_dict:
            args.run_names = params_dict["run_names"]
        else:
            map_paths = sorted(list(map_dir.glob("*.pkl")))
    else:
        map_paths = [map_dir / f"{name}.pkl" for name in args.run_names]
    run_env = args.run_env if "run_env" not in params_dict else params_dict["run_env"]
    run_names = [p.stem for p in map_paths]
    roman_maps = [ROMANMap.from_pickle(str(p)) for p in map_paths]

    # Load ground truth data
    gt_pose_data = []
    for run in run_names:
        os.environ[run_env] = run
        gt_pose_data.append(PoseData.from_dict(params_dict["gt_pose"]))

    # Set up segment matcher
    matcher = SegmentMatcher(SegmentMatchParams.from_yaml(args.params))
    registerer = Registerer(RegisterParams.from_yaml(args.params))

    # Load submap times
    submap_params = SubmapParams.from_yaml(args.params)
    submap_params.creation_method = "set_times"
    submap_times = {}
    for name in run_names:
        submap_dict = json.load(
            (roman_results_dir / "align" / f"{name}_{name}" / f"{name}.sm.json").open(
                "r"
            )
        )
        submap_times[name] = [
            float(sm["seconds"]) + float(sm["nanoseconds"]) * 1e-9
            for sm in submap_dict["submaps"]
        ]

    # Load submaps
    conversion_params = DenseToSparseParams.from_yaml(args.params)
    submap_lists = []
    for name, roman_map in zip(run_names, roman_maps):
        if args.save_general_segments:
            general_segments = DenseToSparseConverter(conversion_params).convert(
                roman_map.segments
            )
            (output_dir / "gsm_maps").mkdir(parents=True, exist_ok=True)
            with (output_dir / "gsm_maps" / f"{name}.pkl").open("wb") as f:
                pickle.dump(general_segments, f, -1)
        submap_params.submap_times = submap_times[name]
        new_sm_list = submaps_from_roman_map(
            roman_map, submap_params, conversion_params
        )
        submap_lists.append(new_sm_list)

    # Run batch submap alignment
    results = batch_submap_align(
        run_names,
        submap_lists,
        gt_pose_data,
        matcher=matcher,
        registerer=registerer,
        output_dir=output_dir,
        roman_maps=roman_maps,
        submap_align_params=ROMANBasedSubmapAlignParams(max_distance=20.0),
    )
