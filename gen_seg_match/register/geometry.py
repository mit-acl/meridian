import numpy as np
import numba
from gen_seg_match.segment.segment_types import SegmentList


def extract_line_arrays(seg_list: SegmentList):
    """Extract line normals (2D) and offsets from a SegmentList.

    Normal = direction rotated +90°.  Offset = dot(normal, point).

    Returns
    -------
    normals : (M, 2) ndarray
    offsets : (M,)  ndarray
    """
    lines = seg_list.get_lines()
    dirs = lines.directions  # (M, 2)
    if len(dirs) == 0:
        return np.empty((0, 2)), np.empty(0)
    normals = np.column_stack([-dirs[:, 1], dirs[:, 0]])
    points = np.array([seg.point for seg in lines]).reshape(-1, 2)
    offsets = np.sum(normals * points, axis=1)
    return normals, offsets


# ---------------------------------------------------------------------------
# Numba-accelerated 2×2 helpers
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def cross_covariance_2x2(p, q, s_norm, t_norm, W_P, W_D):
    """Build 2×2 cross-covariance from centred points + weighted normals."""
    H = np.zeros((2, 2))
    n_p = p.shape[0]
    n_l = s_norm.shape[0]

    if n_p > 0:
        pmx = 0.0; pmy = 0.0; qmx = 0.0; qmy = 0.0
        for j in range(n_p):
            pmx += p[j, 0]; pmy += p[j, 1]
            qmx += q[j, 0]; qmy += q[j, 1]
        pmx /= n_p; pmy /= n_p; qmx /= n_p; qmy /= n_p
        for j in range(n_p):
            pcx = p[j, 0] - pmx; pcy = p[j, 1] - pmy
            qcx = q[j, 0] - qmx; qcy = q[j, 1] - qmy
            H[0, 0] += W_P * pcx * qcx
            H[0, 1] += W_P * pcx * qcy
            H[1, 0] += W_P * pcy * qcx
            H[1, 1] += W_P * pcy * qcy

    for j in range(n_l):
        H[0, 0] += W_D * s_norm[j, 0] * t_norm[j, 0]
        H[0, 1] += W_D * s_norm[j, 0] * t_norm[j, 1]
        H[1, 0] += W_D * s_norm[j, 1] * t_norm[j, 0]
        H[1, 1] += W_D * s_norm[j, 1] * t_norm[j, 1]

    return H


@numba.njit(cache=True)
def svd_rotation_2x2(H):
    """Rotation from 2×2 cross-covariance via SVD.  Returns (R, σ_max)."""
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # if R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0] < 0:
        # Vt[1, 0] *= -1; Vt[1, 1] *= -1
        # R = Vt.T @ U.T
    return R, S[0]


@numba.njit(cache=True)
def solve_translation_2x2(R, p, q, s_norm, s_off, t_off, W_P, W_L_M):
    """Solve 2D translation via 2×2 normal equations.  Returns t (2,)."""
    ATA = np.zeros((2, 2))
    ATb = np.zeros(2)

    for j in range(p.shape[0]):
        rx = R[0, 0] * p[j, 0] + R[0, 1] * p[j, 1]
        ry = R[1, 0] * p[j, 0] + R[1, 1] * p[j, 1]
        bx = q[j, 0] - rx; by = q[j, 1] - ry
        ATA[0, 0] += W_P; ATA[1, 1] += W_P
        ATb[0] += W_P * bx; ATb[1] += W_P * by

    for j in range(s_norm.shape[0]):
        nx = R[0, 0] * s_norm[j, 0] + R[0, 1] * s_norm[j, 1]
        ny = R[1, 0] * s_norm[j, 0] + R[1, 1] * s_norm[j, 1]
        rhs = t_off[j] - s_off[j]
        ATA[0, 0] += W_L_M * nx * nx
        ATA[0, 1] += W_L_M * nx * ny
        ATA[1, 0] += W_L_M * ny * nx
        ATA[1, 1] += W_L_M * ny * ny
        ATb[0] += W_L_M * nx * rhs
        ATb[1] += W_L_M * ny * rhs

    return np.linalg.solve(ATA, ATb)


@numba.njit(cache=True)
def sign_align_normals(R, s_norm, t_norm, t_off):
    """Flip target normals/offsets so they agree with rotated source normals."""
    n_l = s_norm.shape[0]
    t_norm_a = t_norm.copy()
    t_off_a = t_off.copy()
    for j in range(n_l):
        rn0 = R[0, 0] * s_norm[j, 0] + R[0, 1] * s_norm[j, 1]
        rn1 = R[1, 0] * s_norm[j, 0] + R[1, 1] * s_norm[j, 1]
        if rn0 * t_norm[j, 0] + rn1 * t_norm[j, 1] < 0:
            t_norm_a[j, 0] *= -1; t_norm_a[j, 1] *= -1
            t_off_a[j] *= -1
    return t_norm_a, t_off_a


@numba.njit(cache=True)
def point_line_loss_2d(R, t, s_p, t_p, s_norm, t_norm, s_off, t_off,
                       W_P, W_D, W_L_M, bidirectional):
    """Combined 2D point + line registration loss."""
    loss = 0.0
    for j in range(s_p.shape[0]):
        sx = R[0, 0] * s_p[j, 0] + R[0, 1] * s_p[j, 1] + t[0]
        sy = R[1, 0] * s_p[j, 0] + R[1, 1] * s_p[j, 1] + t[1]
        dx = sx - t_p[j, 0]; dy = sy - t_p[j, 1]
        loss += 0.5 * W_P * (dx * dx + dy * dy)

    for j in range(s_norm.shape[0]):
        nx = R[0, 0] * s_norm[j, 0] + R[0, 1] * s_norm[j, 1]
        ny = R[1, 0] * s_norm[j, 0] + R[1, 1] * s_norm[j, 1]
        d_prime = s_off[j] + nx * t[0] + ny * t[1]
        cos_sim = nx * t_norm[j, 0] + ny * t_norm[j, 1]
        if bidirectional and cos_sim < 0:
            cos_sim = -cos_sim
            d_prime = -d_prime
        loss += W_D * (-cos_sim)
        loss += 0.5 * W_L_M * (d_prime - t_off[j]) ** 2

    return loss


# Status codes returned by register_2d_core
REGISTER_OK = 0
REGISTER_NO_ROTATION = 1
REGISTER_NO_TRANSLATION = 2
REGISTER_DEGENERATE_SIGN = 3
REGISTER_SINGULAR = 4

# Shared empty (0, 2) buffer for feasibility calls that pass no points/lines.
_EMPTY_2 = np.empty((0, 2), dtype=np.float64)


@numba.njit(cache=True)
def register_2d_core(p, q, s_norm, s_off, t_norm, t_off,
                     W_P, W_D, W_L_M, eps, dup_eps):
    """Numba-optimized core of Registerer2D.register.

    Performs feasibility checks, initial rotation (with sign disambiguation
    when lines alone determine rotation), final sign-aligned re-solve, and
    returns the inverted SE(2) transform as a 3x3 matrix.

    Returns
    -------
    status : int
        0 on success; non-zero codes described by the REGISTER_* constants.
    T : (3, 3) float64 ndarray
        SE(2) transform mapping target to source (identity on failure).
    """
    num_points = p.shape[0]
    num_lines = s_norm.shape[0]
    T = np.eye(3)

    # Points-only rotation: doubles as the feasibility check (s0 > eps) and,
    # when valid, as the initial R — no need to recompute it below.
    R_pts = np.eye(2)
    points_determine_rotation = False
    if num_points >= 2:
        H_pts = cross_covariance_2x2(p, q, _EMPTY_2, _EMPTY_2, 1.0, 0.0)
        R_pts, s0_pts = svd_rotation_2x2(H_pts)
        points_determine_rotation = s0_pts > eps

    has_rotation = num_lines >= 1 or points_determine_rotation

    # Translation: N has rank 2 iff σ_min(N) > 0.
    has_translation = num_points >= 1
    if not has_translation and num_lines >= 2:
        NN = cross_covariance_2x2(_EMPTY_2, _EMPTY_2, s_norm, s_norm, 0.0, 1.0)
        _, S_NN, _ = np.linalg.svd(NN)
        has_translation = S_NN[1] > eps

    if not has_rotation:
        return REGISTER_NO_ROTATION, T
    if not has_translation:
        return REGISTER_NO_TRANSLATION, T

    if num_lines > 0 and not points_determine_rotation:
        # Lines determine rotation: try both signs for the first line normal,
        # keep the winner's sign-aligned arrays to avoid re-aligning below.
        best_loss = np.inf
        second_best_loss = np.inf
        best_R = np.eye(2)
        best_t_norm_a = t_norm
        best_t_off_a = t_off
        found = False

        for sign_idx in range(2):
            sign_val = -1.0 if sign_idx == 0 else 1.0
            first_t_norm = sign_val * t_norm[0:1]
            H = cross_covariance_2x2(p, q, s_norm[0:1], first_t_norm, W_P, W_D)
            R_cand, s0 = svd_rotation_2x2(H)
            if s0 < eps:
                continue

            t_norm_a, t_off_a = sign_align_normals(R_cand, s_norm, t_norm, t_off)
            trans_cand = solve_translation_2x2(
                R_cand, p, q, s_norm, s_off, t_off_a, W_P, W_L_M
            )

            loss = point_line_loss_2d(
                R_cand, trans_cand, p, q,
                s_norm, t_norm, s_off, t_off,
                W_P, W_D, W_L_M, True,
            )

            if loss < best_loss:
                second_best_loss = best_loss
                best_loss = loss
                best_R = R_cand
                best_t_norm_a = t_norm_a
                best_t_off_a = t_off_a
                found = True
            elif loss < second_best_loss:
                second_best_loss = loss

        if not found or second_best_loss - best_loss < dup_eps:
            return REGISTER_DEGENERATE_SIGN, T
        R = best_R
        t_norm_aligned = best_t_norm_a
        t_off_aligned = best_t_off_a
    else:
        # Points determine rotation — reuse R_pts from the feasibility SVD.
        R = R_pts
        if num_lines > 0:
            t_norm_aligned, t_off_aligned = sign_align_normals(R, s_norm, t_norm, t_off)
        else:
            t_norm_aligned = t_norm
            t_off_aligned = t_off

    # Refine R with the full sign-aligned data.
    if num_lines > 0:
        H_final = cross_covariance_2x2(p, q, s_norm, t_norm_aligned, W_P, W_D)
        R_final, s0_f = svd_rotation_2x2(H_final)
        if s0_f < eps:
            return REGISTER_SINGULAR, T
    else:
        R_final = R

    trans_final = solve_translation_2x2(
        R_final, p, q, s_norm, s_off, t_off_aligned, W_P, W_L_M
    )

    # T = [[R, t], [0, 1]] ; T^-1 = [[R^T, -R^T t], [0, 1]]
    r00 = R_final[0, 0]; r01 = R_final[0, 1]
    r10 = R_final[1, 0]; r11 = R_final[1, 1]
    T[0, 0] = r00; T[0, 1] = r10
    T[1, 0] = r01; T[1, 1] = r11
    T[0, 2] = -(r00 * trans_final[0] + r10 * trans_final[1])
    T[1, 2] = -(r01 * trans_final[0] + r11 * trans_final[1])

    return REGISTER_OK, T
