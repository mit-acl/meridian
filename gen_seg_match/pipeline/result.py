import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial.transform import Rotation as Rot


class AssociationType(Enum):
    POINT_TO_POINT = 1
    LINE_TO_LINE = 2


@dataclass
class PoseEstimationResult:
    associations: tuple = tuple([])
    association_types: tuple = tuple([])
    inlier_ratio: float = np.nan
    T_i_j: np.ndarray = field(default_factory=lambda: np.full((4, 4), np.nan))
    T_i_j_hat: np.ndarray = field(default_factory=lambda: np.full((4, 4), np.nan))
    descriptor_similarity: float = np.nan
    runtime_s: float = np.nan

    @property
    def T_error(self):
        if np.any(np.isnan(self.T_i_j)) or np.any(np.isnan(self.T_i_j_hat)):
            return None
        return np.linalg.inv(self.T_i_j_hat) @ self.T_i_j

    @property
    def translation_error_m(self):
        if np.any(np.isnan(self.T_i_j)) or np.any(np.isnan(self.T_i_j_hat)):
            return np.nan
        return np.linalg.norm((self.T_i_j - self.T_i_j_hat)[:3, 3])

    @property
    def rotation_error_rad(self):
        if self.T_error is None:
            return np.nan
        return Rot.from_matrix(self.T_error[:3, :3]).magnitude()

    @property
    def gt_distance_m(self):
        if np.any(np.isnan(self.T_i_j)):
            return np.nan
        return np.linalg.norm(self.T_i_j[:3, 3])  # translation norm

    @property
    def gt_rotation_diff_rad(self):
        if np.any(np.isnan(self.T_i_j)):
            return np.nan
        return Rot.from_matrix(self.T_i_j[:3, :3]).magnitude()

    @property
    def angle_error_rad(self):
        print("Warning: angle_error_rad is deprecated, use rotation_error_rad instead.")
        return self.rotation_error_rad

    @property
    def num_associations(self):
        return len(self.associations)

    @property
    def num_point_associations(self):
        return sum(
            1 for t in self.association_types if t == AssociationType.POINT_TO_POINT
        )

    @property
    def num_line_associations(self):
        return sum(
            1 for t in self.association_types if t == AssociationType.LINE_TO_LINE
        )


class PoseEstimationResultMatrix(np.ndarray):
    """A numpy ndarray subclass holding PoseEstimationResult objects."""

    def __new__(cls, shape, fill_value: PoseEstimationResult = None):
        assert len(shape) >= 2, "Shape dimension must be at least 2."
        obj = np.empty(shape, dtype=object).view(cls)

        if fill_value is None:
            # Important: create a *new* instance per entry
            for idx in np.ndindex(shape):
                obj[idx] = PoseEstimationResult()
        else:
            for idx in np.ndindex(shape):
                obj[idx] = fill_value

        return obj

    def __array_finalize__(self, obj):
        # Called on new views/slices
        if obj is None:
            return

    @classmethod
    def load(cls, filepath: str):
        """Load a PoseEstimationResultMatrix from a .npz file."""
        data = np.load(filepath, allow_pickle=True)
        results = data["results"]
        return results.view(cls)

    @classmethod
    def concatenate(cls, arrays, axis=0):
        out = np.concatenate(arrays, axis=axis)
        return out.view(cls)

    @property
    def translation_error_m(self):
        """Return a matrix of translation errors."""
        return np.vectorize(lambda r: r.translation_error_m)(self)

    @property
    def angle_error_rad(self):
        """Return a matrix of angle errors."""
        print("Warning: angle_error_rad is deprecated, use rotation_error_rad instead.")
        return self.rotation_error_rad

    @property
    def rotation_error_rad(self):
        """Return a matrix of rotation errors."""
        return np.vectorize(lambda r: r.rotation_error_rad)(self)

    @property
    def num_associations(self):
        """Return a matrix of number of associations."""
        return np.vectorize(lambda r: r.num_associations)(self)

    @property
    def num_point_associations(self):
        """Return a matrix of number of point associations."""
        return np.vectorize(lambda r: r.num_point_associations)(self)

    @property
    def num_line_associations(self):
        """Return a matrix of number of line associations."""
        return np.vectorize(lambda r: r.num_line_associations)(self)

    @property
    def gt_distance_m(self):
        """Return a matrix of ground truth distances."""
        return np.vectorize(lambda r: r.gt_distance_m)(self)

    @property
    def gt_rotation_diff_rad(self):
        """Return a matrix of submap yaw differences."""
        return np.vectorize(lambda r: r.gt_rotation_diff_rad)(self)

    @property
    def descriptor_similarity(self):
        """Return a matrix of descriptor similarities."""
        return np.vectorize(lambda r: r.descriptor_similarity)(self)

    @property
    def has_similarity(self):
        """Check if any result has similarity matrix."""
        return np.any(
            np.vectorize(lambda r: not np.isnan(r.descriptor_similarity))(self)
        )

    @property
    def runtime_s(self):
        """Return a matrix of runtimes."""
        return np.vectorize(lambda r: r.runtime_s)(self)

    def save(self, filepath: str):
        """Save the PoseEstimationResultMatrix to a .npz file."""
        np.savez_compressed(filepath, results=self)

    def plot(
        self,
        dpi: int = 250,
        dist_thresh: float = 5.0,
        angle_thresh_deg: float = 10.0,
        gt_patches=None,
    ):
        show_sim = self.has_similarity

        fig, ax = plt.subplots(3, 2, figsize=(8, 12), dpi=dpi)
        fig.subplots_adjust(wspace=0.3)
        # TODO: add suptitle
        # fig.suptitle(f"{results.submap_io.run_name}: {results.submap_io.robot_names[0]}, {results.submap_io.robot_names[1]}")

        mp = ax[0, 0].imshow(self.gt_distance_m, cmap="magma", vmin=0)
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        ax[0, 0].set_title("Submaps Center Distance (m)")

        mp = ax[0, 1].imshow(
            np.rad2deg(self.gt_rotation_diff_rad), cmap="magma", vmin=0
        )
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        ax[0, 1].set_title("Submap Rotation Difference (deg)")

        rotation_error_mat = np.rad2deg(self.rotation_error_rad.copy())
        dist_error_mat = self.translation_error_m.copy()
        rotation_error_mat[
            np.bitwise_and(
                dist_error_mat > dist_thresh,
                np.bitwise_not(np.isnan(rotation_error_mat)),
            )
        ] = angle_thresh_deg
        dist_error_mat[
            np.bitwise_and(
                rotation_error_mat > angle_thresh_deg,
                np.bitwise_not(np.isnan(dist_error_mat)),
            )
        ] = dist_thresh

        mp = ax[1, 0].imshow(
            dist_error_mat, cmap="viridis_r", vmax=dist_thresh, vmin=0.0
        )
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        ax[1, 0].set_title("Registration Translation Error (m)")

        mp = ax[1, 1].imshow(
            rotation_error_mat, cmap="viridis_r", vmax=angle_thresh_deg, vmin=0.0
        )
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        ax[1, 1].set_title("Registration Rotation Error (deg)")

        mp = ax[2, 0].imshow(self.num_associations, cmap="viridis", vmin=0)
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        ax[2, 0].set_title("Number of Associations")

        if show_sim:
            mp = ax[2, 1].imshow(
                self.descriptor_similarity, cmap="viridis", vmin=0.0, vmax=1.0
            )
            fig.colorbar(mp, fraction=0.04, pad=0.04)
            ax[2, 1].set_title("Similarity Score")

        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i, j].set_xlabel("submap index (robot 2)")
                ax[i, j].set_ylabel("submap index (robot 1)")
                ax[i, j].grid(True)

        if gt_patches:
            for a in ax.flat:
                if not a.has_data():
                    continue
                for i_a, j_a in gt_patches:
                    rect = Rectangle(
                        (j_a - 0.5, i_a - 0.5),
                        1,
                        1,
                        linewidth=2,
                        edgecolor="lime",
                        facecolor="none",
                    )
                    a.add_patch(rect)

        if not show_sim:
            fig.delaxes(ax[2, 1])
        return fig, ax

    def plot_point_vs_line_associations(self):
        """Plot number of point vs line associations for each result."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        mp = ax[0].imshow(self.num_point_associations, cmap="viridis", vmin=0)
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        mp = ax[1].imshow(self.num_line_associations, cmap="viridis", vmin=0)
        fig.colorbar(mp, fraction=0.04, pad=0.04)
        ax[0].set_title("Number of Point Associations")
        ax[1].set_title("Number of Line Associations")
        return fig, ax
