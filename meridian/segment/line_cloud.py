import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from dataclasses import dataclass
from robotdatapy.transform import transform


@dataclass
class LineCloud:
    lines: np.ndarray

    def __post_init__(self):
        self.lines = np.array(self.lines)

    def __len__(self):
        return len(self.lines)

    def __eq__(self, other):
        if not isinstance(other, LineCloud):
            return False
        if len(self.lines) != len(other.lines):
            return False
        for line1, line2 in zip(self.lines, other.lines):
            if not np.array_equal(line1, line2):
                return False
        return True

    def centers_and_directions(self):
        centers = self.centers()
        directions = np.array(
            [
                (line[1] - line[0]) / np.linalg.norm(line[1] - line[0])
                for line in self.lines
            ]
        )
        return np.hstack((centers, directions))

    def centers(self):
        return np.array([(line[0] + line[1]) / 2.0 for line in self.lines])

    def copy_and_transform(self, T):
        new_lines = transform(T, self.lines.reshape(-1, 2)).reshape(-1, 2, 2)
        return LineCloud(new_lines)

    def plot(self, color="random", ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        for line in self.lines:
            p1, p2 = line
            line_color = (
                color
                if color != "random"
                else np.random.rand(
                    3,
                )
            )
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=line_color, linewidth=2)
        ax.axis("equal")


@dataclass
class LineMergerParams:
    line_len: float = 2.0
    min_merge_angle_deg: float = 10.0
    min_merge_dist: float = 1.0
    max_num_iterations: int = 1


class LineMerger:
    def __init__(self, params: LineMergerParams):
        self.params = params

    def merge_lines(self, line_cloud: LineCloud, verbose: bool = True) -> LineCloud:
        current_line_cloud = deepcopy(line_cloud)
        previous_line_cloud = deepcopy(line_cloud)

        for it in range(self.params.max_num_iterations):
            if verbose:
                print(f"iteration {it}")
            # Step 1: break apart any lines longer than line_len
            correct_line_lens = []

            for line in current_line_cloud.lines:
                p1, p2 = deepcopy(line)
                line_len = np.linalg.norm(p2 - p1)
                # if 150% of desired line length then split it
                if line_len > self.params.line_len * 1.5:
                    num_segments = int(np.ceil(line_len / self.params.line_len))
                    for i in range(num_segments):
                        new_p1 = p1 + (p2 - p1) * (i / num_segments)
                        new_p2 = p1 + (p2 - p1) * ((i + 1) / num_segments)
                        correct_line_lens.append([new_p1, new_p2])
                else:
                    center = (p1 + p2) / 2.0
                    dir_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
                    new_p1 = center - dir_vec * self.params.line_len / 2.0
                    new_p2 = center + dir_vec * self.params.line_len / 2.0
                    correct_line_lens.append([new_p1, new_p2])

            # Step 2: merge lines that are close and have similar orientation
            untouched_lines = deepcopy(correct_line_lens)
            checked_lines = []

            while len(untouched_lines) > 0:
                curr_line = untouched_lines.pop(0)
                p1, p2 = curr_line
                center_p = (p1 + p2) / 2.0
                dir_p = (p2 - p1) / np.linalg.norm(p2 - p1)

                merged = False
                for i, candidate_line in enumerate(untouched_lines):
                    q1, q2 = candidate_line
                    center_q = (q1 + q2) / 2.0
                    dir_q = (q2 - q1) / np.linalg.norm(q2 - q1)

                    cosine_sim = np.abs(np.dot(dir_p, dir_q))
                    angle = np.rad2deg(np.arccos(cosine_sim))

                    if angle < self.params.min_merge_angle_deg:
                        # dists = [
                        #     np.linalg.norm(p1 - q1),
                        #     np.linalg.norm(p1 - q2),
                        #     np.linalg.norm(p2 - q1),
                        #     np.linalg.norm(p2 - q2),
                        # ]
                        # min_dist = min(dists)
                        if (
                            np.linalg.norm(center_p - center_q)
                            < self.params.min_merge_dist
                        ):
                            untouched_lines.pop(i)
                            mean_dir_unnormalized_option_1 = dir_p + dir_q
                            mean_dir_unnormalized_option_2 = dir_p - dir_q
                            if np.linalg.norm(
                                mean_dir_unnormalized_option_1
                            ) > np.linalg.norm(mean_dir_unnormalized_option_2):
                                mean_dir = mean_dir_unnormalized_option_1
                            else:
                                mean_dir = mean_dir_unnormalized_option_2
                            mean_dir /= np.linalg.norm(mean_dir)
                            center = (center_p + center_q) / 2.0
                            new_p1 = center - mean_dir * self.params.line_len / 2.0
                            new_p2 = center + mean_dir * self.params.line_len / 2.0
                            curr_line = [new_p1, new_p2]
                            # new_p1 = p1 if dists[0] < dists[1] else p2 if dists[2] < dists[3] else q1
                            # new_p2 = q2 if dists[0] < dists[2] else q1 if dists[1] < dists[3] else p2
                            # curr_line = [new_p1, new_p2]
                            checked_lines.append(curr_line)
                            merged = True
                            break
                            # p1, p2 = curr_line
                            # center = (p1 + p2) / 2.0
                            # dir1 = (p2 - p1) / np.linalg.norm(p2 - p1)
                if not merged:
                    checked_lines.append(curr_line)

            current_line_cloud = LineCloud(checked_lines)
            if current_line_cloud == previous_line_cloud:
                if verbose:
                    print(f"no changes in iteration {it}, stopping early")
                break
            previous_line_cloud = deepcopy(current_line_cloud)
        if verbose:
            print(f"final number of lines: {len(current_line_cloud.lines)}")
        return current_line_cloud
