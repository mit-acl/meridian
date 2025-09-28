import numpy as np
from typing import List
import matplotlib.pyplot as plt
import clipperpy
from dataclasses import dataclass

from gen_seg_match.segment.segment_types import SegmentPoint, SegmentLine, \
    SegmentPlane, GeneralSegment

class InsufficientAssociationsException(Exception):
    
    def __init__(self, map1_len, map2_len, n_associations=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)

@dataclass
class SegmentMatcherParams:

    dim: int = 3;                                   # dimension of points (2 or 3)
    ratio_feature_dim: int = 0;                     # number of ratio features (e.g., volume)
    cos_feature_dim: int = 0;                       # number of features used for cosine similarity
    sigma_dist: float = 0.4;                        # spread / "variance" of exponential kernel
    epsilon_dist: float = 0.6;                      # bound on consistency score, determines if inlier/outlier
    min_dist: float = 0.0;                          # minimum allowable distance between inlier points in the same dataset
    sigma_angle_rad: float = np.deg2rad(10.0);      # spread / "variance" of exponential kernel
    epsilon_angle_rad: float = np.deg2rad(20.0);    # bound on consistency score, determines if inlier/outlier
    min_angle_rad: float = 0.0;                     # minimum allowable angle (in radians) between inlier segments in the same dataset
    distance_weight: float = 1.0;                   # weight of pairwise similarity in single/pairwise fusion
    ratio_weight: float = 1.0;                      # weight of cosine similarity in single similarity fusion
    cosine_weight: float = 1.0;                     # weight of cosine similarity in single similarity fusion
    ratio_epsilon: np.ndarray = np.zeros(0);        # bound on feature ratio score, determines if inlier/outlier
    cosine_min: float = 0.5;                        # cosine similarity scaled so that cosine_min maps to 0.0 similarity score
    cosine_max: float = 0.7;                        # cosine similarity scaled so that cosine_max maps to 1.0 similarity score
    gravity_guided: bool = False;                   # whether to use gravity-guided prior
    gravity_unc_ang_rad: float = 0.0;               # uncertainty adjustment for gravity direction in radians

    def to_clipper(self,):
        iparams = clipperpy.invariants.GeneralSegmentDistanceParams()
        iparams.dim = self.dim
        iparams.ratio_feature_dim = self.ratio_feature_dim
        iparams.cos_feature_dim = self.cos_feature_dim
        iparams.sigma_dist = self.sigma_dist
        iparams.epsilon_dist = self.epsilon_dist
        iparams.min_dist = self.min_dist
        iparams.sigma_angle_rad = self.sigma_angle_rad
        iparams.epsilon_angle_rad = self.epsilon_angle_rad
        iparams.min_angle_rad = self.min_angle_rad
        iparams.distance_weight = self.distance_weight
        iparams.ratio_weight = self.ratio_weight
        iparams.cosine_weight = self.cosine_weight
        iparams.ratio_epsilon = self.ratio_epsilon
        iparams.cosine_min = self.cosine_min
        iparams.cosine_max = self.cosine_max
        iparams.gravity_guided = self.gravity_guided
        iparams.gravity_unc_ang_rad = self.gravity_unc_ang_rad
        return iparams

class SegmentMatcher():

    def __init__(self, params: SegmentMatcherParams):
        self.params = params

    # def match(self, map1: List[GeneralSegment], map2: List[GeneralSegment]):
    def match(self, points1: List[GeneralSegment], lines1: List[GeneralSegment],
              points2: List[GeneralSegment], lines2: List[GeneralSegment]):
        if len(points1) + len(lines1) == 0 or len(points2) + len(lines2) == 0:
            return np.array([[]])
        # if len(map1) == 0 or len(map2) == 0:
        #     return np.array([[]])
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, points1, lines1, points2, lines2)
        clipper.solve()
        Ain = clipper.get_selected_associations()
        return Ain
    
    def _setup_solver(self):
        invariant = clipperpy.invariants.GeneralSegmentDistance(self.params.to_clipper())
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)
        return clipper
    
    # def _setup_problem(self, clipper, map1: list, map2: list):
    def _setup_problem(self, clipper, 
            points1: List[GeneralSegment], lines1: List[GeneralSegment],
            points2: List[GeneralSegment], lines2: List[GeneralSegment]):
        # points1 = [obj for obj in map1 if type(obj) == SegmentPoint]
        # lines1 = [obj for obj in map1 if type(obj) == SegmentLine]
        # points2 = [obj for obj in map2 if type(obj) == SegmentPoint]
        # lines2 = [obj for obj in map2 if type(obj) == SegmentLine]

        # set up all to all matching between points and lines separately
        A_init_points = clipperpy.utils.create_all_to_all(len(points1), len(points2))
        A_init_lines = clipperpy.utils.create_all_to_all(len(lines1), len(lines2))
        A_init_lines[:,0] += len(points1)
        A_init_lines[:,1] += len(points2)
        A_init = np.vstack([A_init_points, A_init_lines])

        map1_arrays = [obj.to_array() for obj in points1] + [obj.to_array() for obj in lines1]
        map2_arrays = [obj.to_array() for obj in points2] + [obj.to_array() for obj in lines2]
        max_d = max([arr.shape[0] for arr in map1_arrays + map2_arrays])

        map1_arrays = [np.pad(arr, (0, max_d - arr.shape[0]), 'constant', constant_values=0.0) for arr in map1_arrays]
        map2_arrays = [np.pad(arr, (0, max_d - arr.shape[0]), 'constant', constant_values=0.0) for arr in map2_arrays]
        map1_cl = np.array(map1_arrays)
        map2_cl = np.array(map2_arrays)

        clipper.score_pairwise_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init
    
    def get_MCA(self, points1: List[GeneralSegment], lines1: List[GeneralSegment],
              points2: List[GeneralSegment], lines2: List[GeneralSegment]):
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, points1, lines1, points2, lines2)
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return M, C, A_init
    
    def match_multiple(self, points1: List[GeneralSegment], lines1: List[GeneralSegment],
              points2: List[GeneralSegment], lines2: List[GeneralSegment], num_solutions=2):
        M, C, A = self.get_MCA(points1, lines1, points2, lines2)
        M_orig = M.copy()
        clipper = clipperpy.CLIPPER(clipperpy.invariants.PairwiseInvariant(), clipperpy.Params())
        solutions = []

        for k in range(num_solutions):
            clipper.set_matrix_data(M=M, C=C)
            clipper.solve()

            solution_nodes = clipper.get_solution().nodes
            Ain = np.zeros((len(solution_nodes), 2)).astype(np.int64)
            for i in range(len(solution_nodes)):
                Ain[i,:] = A[solution_nodes[i],:]
            
            u_sol = clipper.get_solution().u.copy()
            for i in range(u_sol.shape[0]):
                u_sol[i] = u_sol[i] if i in solution_nodes else 0.0
            if len(solution_nodes) == 0:
                score = 0
            else:
                score = u_sol.T @ M_orig @ u_sol / (u_sol.T @ u_sol)
            solutions.append((Ain.copy(), score))

            if k + 1 < num_solutions:
                row_indices, col_indices = np.meshgrid(solution_nodes, solution_nodes, indexing='ij')
                if len(row_indices) != 0 and len(col_indices) != 0:
                    M[row_indices,col_indices] = 0.0

        return solutions

    # def T_align(self, map1: List[Object], map2: List[Object], correspondences: np.array = None):
    #     """
    #     Computes the transformation that aligns map2 to map1.

    #     Args:
    #         map1 (List[Object]): Object list in frame 1
    #         map2 (List[Object]): Object list in frame 2
    #         correspondences (np.array, shape=(n,2), optional): If correspondences have already 
    #             been found, set to None. Otherwise, performs register before aligning. Aligns using 
    #             Arun's method. Defaults to None.

    #     Returns:
    #         np.array: Transformation matrix that aligns map2 to map1
    #     """
    #     if len(map1) == 0 or len(map2) == 0:
    #         raise InsufficientAssociationsException(len(map1), len(map2))

    #     if correspondences is None:
    #         correspondences = self.register(map1, map2)
    #     if len(correspondences) < self.dim:
    #         raise InsufficientAssociationsException(len(map1), len(map2), len(correspondences))

    #     pts1 = np.array([map1[corr[0]].center.reshape(-1)[:self.dim] for corr in correspondences])
    #     pts2 = np.array([map2[corr[1]].center.reshape(-1)[:self.dim] for corr in correspondences])

    #     weights = np.ones((pts1.shape[0],1))
    #     weights = weights.reshape((-1,1))
    #     mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
    #     mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
    #     pts1_mean_reduced = pts1 - mean1
    #     pts2_mean_reduced = pts2 - mean2
    #     assert pts1_mean_reduced.shape == pts2_mean_reduced.shape
    #     H = pts1_mean_reduced.T @ (pts2_mean_reduced * weights)
    #     U, s, Vh = np.linalg.svd(H)
    #     R = U @ Vh
    #     if np.allclose(np.linalg.det(R), -1.0):
    #         Vh_prime = Vh.copy()
    #         Vh_prime[-1,:] *= -1.0
    #         R = U @ Vh_prime
    #     t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
    #     T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
    #     return T
    
    # def view_registration(self, map1: List[Object], map2: List[Object], correspondences: np.array, T: np.array, ax=None, **kwargs):
    #     """
    #     Visualize the registration between map1 and map2

    #     Args:
    #         map1 (List[Object]): Object list in frame 1
    #         map2 (List[Object]): Object list in frame 2
    #         correspondences (np.array, shape=(n,2)): Correspondences between map1 and map2
    #         T (np.array): Transformation matrix that aligns map2 to map1
    #     """
    #     if ax is None:
    #         _, ax = plt.subplots()

    #     map2_cp = [obj.copy() for obj in map2]
    #     for obj in map2_cp:
    #         obj.transform(T)

    #     for obj in map1:
    #         obj.plot2d(ax, color='maroon', **kwargs)

    #     for obj in map2_cp:
    #         obj.plot2d(ax, color='blue', **kwargs)

    #     for corr in correspondences:
    #         ax.plot([map1[corr[0]].centroid[0], map2_cp[corr[1]].centroid[0]], 
    #                  [map1[corr[0]].centroid[1], map2_cp[corr[1]].centroid[1]], 
    #                  color='lawngreen', linestyle='dotted')
        
    #     ax.set_aspect('equal')
    #     return ax