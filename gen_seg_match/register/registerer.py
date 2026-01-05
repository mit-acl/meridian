import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations, product

from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
    SegmentPlane,
)
from gen_seg_match.params import RegisterParams
from gen_seg_match.register.geometry import cross_covariance, rank, skew
from gen_seg_match.register.optimization import PointLineLoss, PointLineLoss_numpy

class InsufficientAssociationsException(Exception):
    def __init__(self, map1_len, map2_len, n_associations=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)

@dataclass
class RegistrationResult:

    transformation: np.ndarray
    losses: List[float]

class Registerer:
    
    def __init__(self, params: RegisterParams):
        self.params = params
        
    def register(
        self,
        source: List[GeneralSegment],
        target: List[GeneralSegment],
        gravity_dir1: np.ndarray = None,
        gravity_dir2: np.ndarray = None,
        correspondences: np.array = None,
    ):
        """
        Computes the transformation that aligns target to source (T^source_target). Currently only uses SegmentPoint correspondences.

        Args:
            source (List[GeneralSegment]): Segment list in frame 1
            target (List[GeneralSegment]): Segment list in frame 2
            correspondences (np.array, shape=(n,2), optional): If correspondences have already
                been found, set to None. Otherwise, performs register before aligning. Defaults to None.

        Returns:
            np.array: Transformation matrix that aligns target to source (T^source_target).
        """
        if len(source) == 0 or len(target) == 0 or len(correspondences) == 0:
            raise InsufficientAssociationsException(len(source), len(target))

        if correspondences is None:
            assert len(source) == len(target)
            correspondences = np.array([[seg1.id, seg2.id] for seg1, seg2 in zip(source, target)])

        source = SegmentList(source).sublist_from_ids(correspondences[:, 0])
        target = SegmentList(target).sublist_from_ids(correspondences[:, 1])

        for i in range(len(correspondences)):
            assert type(source[i]) == type(target[i]), "Corresponded segments must be of the same type"
        
        num_points = len(source.get_points())
        num_lines = len(source.get_lines())
        num_planes = len(source.get_planes())

        p = source.get_points().points
        q = target.get_points().points

        l_dirs, l_mom = source.get_lines().directions, source.get_lines().moments
        j_dirs, j_mom = target.get_lines().directions, target.get_lines().moments

        H = cross_covariance(
            p,
            q,
            [],
            [],
            W_P=self.params.point_weight,
            W_L_D=self.params.line_direction_weight,
        )

        # TODO: may be overly optimistic about the number of lines needed
        lines_needed = max(2 - rank(H, tol=self.params.lin_eps), 0)
        if num_points == 0:
            lines_needed = 3

        assert num_lines >= lines_needed, f"At least {lines_needed} line correspondences are required to solve for rotation."

        line_idxs = []
        if lines_needed > 0:
            for idx_comb in combinations(range(num_lines), lines_needed):
                l_dir_subset = l_dirs[list(idx_comb)]
                j_dir_subset = j_dirs[list(idx_comb)]

                H_temp = H + cross_covariance([], [], l_dir_subset, j_dir_subset, self.params.point_weight, self.params.line_direction_weight)

                if rank(H_temp, tol=self.params.lin_eps) >= 2:
                    line_idxs = list(idx_comb)
                    break
        
        assert len(line_idxs) == lines_needed, "Could not find a sufficient set of non-degenerate line correspondences to solve for rotation."

        if lines_needed > 0:
            PLS = PointLineLoss(
                W_P=self.params.point_weight,
                W_L_D=self.params.line_direction_weight,
                W_L_M=self.params.line_moment_weight,
                bidirectional=True
            ).to(self.params.device)

            best_loss, second_best_loss = float('inf'), float('inf')
            best_R = None

            source_subset = source.get_points() + [source.get_lines()[line_idx_i] for line_idx_i in line_idxs]

            for signs in product([-1, 1], repeat=lines_needed):
                j_subset_signed = [target.get_lines()[line_idx_i].copy() for line_idx_i in line_idxs]
                for line_idx, line in enumerate(j_subset_signed):
                    line.direction *= signs[line_idx]
                target_subset_signed = target.get_points() + j_subset_signed
                
                R_candidate, t_candidate = self.aruns_extended(source_subset, target_subset_signed, rotation_only=False)

                loss = PointLineLoss_numpy(PLS, R_candidate, t_candidate, source, target, device=self.params.device)

                if loss < best_loss:
                    second_best_loss = best_loss                    
                    best_loss = loss
                    best_R = R_candidate
                elif loss < second_best_loss:
                    second_best_loss = loss

            assert (second_best_loss - best_loss) > self.params.dup_eps, \
                "Degenerate configuration: multiple sign combinations yield similar losses."

            R = best_R

        else:
            R, _ = self.aruns_extended(source.get_points(), target.get_points(), rotation_only=True)

        
        # Make target line directions sign-consistent with source
        target_consistent = self.compute_consistent_line_directions(R, source, target)

        return self.register_consistent(source, target_consistent)

    def register_consistent(self, source: SegmentList, target: SegmentList) -> RegistrationResult:
        R, t = self.aruns_extended(source, target, rotation_only=False)
        
        if self.params.run_gd:
            assert False, "Refinement not yet supported"
            R, t, losses = refine_transform(R, t, source, target, device=self.params.device, **self.params.refine_transforms_kwargs)
        else:
            losses = []

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T = np.linalg.inv(T)

        return RegistrationResult(transformation=T, losses=losses)
    
    def compute_consistent_line_directions(self, R: np.ndarray, source: SegmentList, target: SegmentList) -> SegmentList:
        """
        Create a sign-consistent version of the target cloud by flipping line directions
        and moments together to align with the rotated source lines.
        
        Parameters:
        R : np.ndarray
            A 3x3 rotation matrix estimate.
        source : SegmentList
            Source point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
        target : SegmentList
            Target point-line cloud with points (Nx3) and sign-ambiguous lines (Mx6 (d, m) Plücker coordinates).
        
        Returns:
        target_consistent : SegmentList
            Target cloud with sign-consistent line representations.
        """
        l_dir = source.get_lines().directions
        j_lines = target.get_lines().copy()
        
        for i in range(len(l_dir)):
            l_dir_rotated = R @ l_dir[i]
            if np.dot(l_dir_rotated, j_lines[i].direction) < 0:
                # Flip direction
                j_lines[i].direction = -j_lines[i].direction

        return SegmentList(target.get_points() + j_lines)

    
    
    
    def aruns_extended(self, source: SegmentList, target: SegmentList, 
                   rotation_only: Optional[bool]=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the transformation between associated point-line clouds
        using Arun's method extended to include lines.
        
        Parameters:
        source : SegmentList
            Source point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
        target : SegmentList
            Target point-line cloud (sign-consistent) with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
        rotation_only : Optional[bool]
            If True, only compute rotation and set translation to zero.

        Returns:
        R : np.ndarray
            A 3x3 rotation matrix.
        t : np.ndarray
            A 3x1 translation vector.
        """
        
        num_points, num_lines = len(source.get_points()), len(source.get_lines())
        assert num_points == len(target.get_points()) and num_lines == len(target.get_lines()), "Source and target must have the same number of points and lines."
        assert num_lines >= 2 or num_points + num_lines >= 3, "At least two line or three total correspondences are required."
        
        p, q = source.get_points().points, target.get_points().points
        l_dir, j_dir = source.get_lines().directions, target.get_lines().directions

        H = cross_covariance(p, q, l_dir, j_dir, W_P=self.params.point_weight, W_L_D=self.params.line_direction_weight)
        
        U, S, Vt = np.linalg.svd(H)
            
        if np.abs(S[1]) < self.params.lin_eps:
            raise ValueError('Degenerate configuration: second singular value is too small')
        
        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[2, :] *= -1
        
        R = Vt.T @ U.T
        t = self.solve_translation(R, source, target) if not rotation_only else np.zeros(3)
                    
        return R, t

    def solve_translation(self, R: np.ndarray, source: SegmentList, target: SegmentList) -> np.ndarray:
        """
        Solve for the translation vector given rotation and associated point-line clouds
        by minimizing L2 error.
        
        Parameters:
        R : np.ndarray
            A 3x3 rotation matrix.
        source : PointLineCloud
            Source point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
        target : PointLineCloud
            Target point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
        
        Returns:
        t : np.ndarray
            A 3x1 translation vector.
        """
        num_points, num_lines = len(source.get_points()), len(source.get_lines())
        assert num_points == len(target.get_points()) and num_lines == len(target.get_lines()), "Source and target must have the same number of points and lines."
        assert num_points + num_lines > 0, "At least one correspondence is required to solve for translation."
        
        p, q = source.get_points().points, target.get_points().points
        l_dir, l_mom = source.get_lines().directions, source.get_lines().moments
        j_dir, j_mom = target.get_lines().directions, target.get_lines().moments
        
        A = np.zeros((3 * (num_points + num_lines), 3))
        b = np.zeros(3 * (num_points + num_lines))

        W_P_SQ = np.sqrt(self.params.point_weight)
        W_L_M_SQ = np.sqrt(self.params.line_moment_weight)
        
        # TODO: optimize with vectorization

        # Point constraints: t = q - R @ p
        for i in range(num_points):
            A[3*i:3*i+3] = W_P_SQ * np.eye(3)
            b[3*i:3*i+3] = W_P_SQ * (q[i] - R @ p[i])
        
        # Line constraints: [j_dir]_x @ t = j_mom - R @ l_mom
        for j in range(num_lines):
            A[3*(num_points+j):3*(num_points+j)+3] = W_L_M_SQ * -skew(j_dir[j]) # TODO: should be R @ l_dir[j]?
            b[3*(num_points+j):3*(num_points+j)+3] = W_L_M_SQ * (j_mom[j] - R @ l_mom[j])
        
        t, *_ = np.linalg.lstsq(A, b, rcond=None)
        return t