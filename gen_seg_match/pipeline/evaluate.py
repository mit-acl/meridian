import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict

from gen_seg_match.pipeline.result import PoseEstimationResultMatrix
from gen_seg_match.utils import expandvars_recursive

STANDARD_YAW_DIFFS = {
    "all": (0.0, 180.0),
    "0 deg": (0.0, 60.0),
    "90 deg": (60.0, 120.0),
    "180 deg": (120.0, 180.0),
}


@dataclass
class EvalParams:
    angular_err_thresh_deg: float = 5.0
    distance_err_thresh_m: float = 1.0
    evaluation_distance_m: float = 10.0
    robot_names: List[str] = None
    include_inter_robot: bool = False

    def __post_init__(self):
        assert not self.include_inter_robot, (
            "Inter-robot evaluation not implemented yet."
        )

    @property
    def robot_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for i in range(len(self.robot_names)):
            start_idx = i if self.include_inter_robot else i + 1
            for j in range(start_idx, len(self.robot_names)):
                pairs.append((self.robot_names[i], self.robot_names[j]))
        return pairs

    @property
    def robot_pairs_as_strings(self) -> List[str]:
        return [f"{r1}_{r2}" for r1, r2 in self.robot_pairs]


@dataclass
class EvalInput:
    directory: str
    name: str = None
    map_directory: str = None
    params_directory: str = None

    def get_directory(self):
        assert os.path.isdir(self.directory), (
            f"Directory {self.directory} does not exist."
        )
        if os.path.isdir(f"{self.directory}/align"):
            return f"{self.directory}/align"
        return self.directory

    def get_name(self):
        if self.name is not None:
            return self.name
        return os.path.basename(os.path.normpath(self.directory))

    def get_map_dir(self):
        return self._get_dir("map", self.map_directory)

    def _get_dir(self, basename: str, first_try: str = None):
        if first_try is not None:
            assert os.path.isdir(first_try), f"Directory {first_try} does not exist."
            return first_try
        dir_option1 = f"{self.directory}/{basename}"
        dir_option2 = f"{self.directory}/../{basename}"
        assert os.path.isdir(dir_option1) or os.path.isdir(dir_option2), (
            f"Directory does not exist in either {dir_option1} or {dir_option2}."
        )
        if os.path.isdir(dir_option1):
            return dir_option1
        return dir_option2


class SubmapAlignEvaluator:
    def __init__(self, params: EvalParams):
        self.params = params
        self.results: Dict[str, PoseEstimationResultMatrix] = {}

    def load_results(self, eval_inputs: Union[EvalInput, List[EvalInput]]):
        if isinstance(eval_inputs, EvalInput):
            eval_inputs = [eval_inputs]
        for eval_input in eval_inputs:
            result_paths = self._get_results_paths(eval_input)
            combined_results = []
            for path, robots in zip(result_paths, self.params.robot_pairs):
                combined_results.append(
                    PoseEstimationResultMatrix.load(path).reshape(-1)
                )

            self.results[eval_input.get_name()] = (
                PoseEstimationResultMatrix.concatenate(combined_results)
            )

    def evaluate_align_success_rate(
        self,
        yaw_diff_min_deg: float = 0.0,
        yaw_diff_max_deg: float = 180.0,
    ) -> Dict[str, float]:
        success_rates = {}
        for name, results in self.results.items():
            relevant_alignments = (
                (results.gt_distance_m <= self.params.evaluation_distance_m)
                & (results.gt_rotation_diff_rad >= np.deg2rad(yaw_diff_min_deg))
                & (results.gt_rotation_diff_rad <= np.deg2rad(yaw_diff_max_deg))
            )
            correct_alignments = (
                results.translation_error_m <= self.params.distance_err_thresh_m
            ) & (
                results.angle_error_rad
                <= np.deg2rad(self.params.angular_err_thresh_deg)
            )
            num_relevant = np.nansum(relevant_alignments)
            num_correct = np.nansum(relevant_alignments & correct_alignments)
            success_rate = (
                num_correct / num_relevant if num_relevant > 0 else float("nan")
            )
            success_rates[name] = success_rate
        return success_rates

    # def evaluate_timing(self) -> Dict[str, float]:
    #     timing_results = {}
    #     for name, results in self.results.items():
    #         if results.timing_list is None:
    #             timing_results[name] = float('nan')
    #             continue
    #         mean_time = np.nanmean(results.timing_list)
    #         timing_results[name] = mean_time
    #     return timing_results

    def _get_results_paths(self, eval_input: EvalInput) -> List[str]:
        dir_path = eval_input.get_directory()
        result_files = []
        for robot_pair in self.params.robot_pairs_as_strings:
            file_path = os.path.join(dir_path, robot_pair, "results.npz")
            assert os.path.isfile(file_path), f"Result file {file_path} does not exist."
            result_files.append(file_path)

        return result_files

    def _get_named_results(
        self, names: List[str] = None
    ) -> Dict[str, PoseEstimationResultMatrix]:
        if names is None:
            return self.results
        else:
            return {name: self.results[name] for name in names if name in self.results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Submap Alignment Results")
    parser.add_argument(
        "-s",
        "--eval-all-subdirs",
        nargs="+",
        type=str,
        default=[],
        help="Evaluate all subdirectories in the given directories.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="Input directory (should be root of the ROMAN demo output)."
        + "Optionally, include a name for the method."
        + "Example: -i /path/to/output1 -i /path/to/output2 cooloer method",
    )
    parser.add_argument(
        "-r",
        "--robots",
        nargs="+",
        required=True,
        type=str,
        help="Names of the robots involved in the alignment.",
    )
    parser.add_argument(
        "-d",
        "--eval-dist",
        type=float,
        default=10.0,
        help="Distance threshold for evaluation (in meters).",
    )
    args = parser.parse_args()

    eval_params = EvalParams(
        robot_names=args.robots,
        evaluation_distance_m=args.eval_dist,
    )

    for input_entry in args.input:
        assert len(input_entry) <= 2, (
            "Input entry must be a directory and optionally a name for the method."
        )

    evaluator = SubmapAlignEvaluator(eval_params)

    eval_inputs = []
    for input_entry in args.input:
        dir_path = expandvars_recursive(input_entry[0])
        name = input_entry[1] if len(input_entry) == 2 else None
        eval_inputs.append(EvalInput(directory=dir_path, name=name))

    for base_dir in args.eval_all_subdirs:
        for subdir in os.listdir(base_dir):
            dir_path = expandvars_recursive(os.path.join(base_dir, subdir))
            if os.path.isdir(dir_path):
                eval_inputs.append(EvalInput(directory=dir_path))

    evaluator.load_results(eval_inputs)

    success_rates = {}
    for yaw_diff_label, (min_deg, max_deg) in STANDARD_YAW_DIFFS.items():
        success_rates[yaw_diff_label] = evaluator.evaluate_align_success_rate(
            yaw_diff_min_deg=min_deg, yaw_diff_max_deg=max_deg
        )

    # timing_results = evaluator.evaluate_timing()

    for yaw_diff_label, rates in success_rates.items():
        print(f"\n=== Alignment Success Rates for Yaw Diff: {yaw_diff_label} ===")
        for name, rate in rates.items():
            print(f"{name}: {rate:.3f}")

    # print(f"\n=== Timing Results (ms) ===")
    # for name, time in timing_results.items():
    #     print(f"{name}: {time*1e3:.1f}")


if __name__ == "__main__":
    main()
