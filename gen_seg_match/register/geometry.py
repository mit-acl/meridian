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
    if R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0] < 0:
        Vt[1, 0] *= -1; Vt[1, 1] *= -1
        R = Vt.T @ U.T
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


@numba.njit(cache=True)
def register_single_2x2(p, q, ns, nt, os_, ot, eps, W_P, W_D, W_L_M):
    """Register a single 2D problem.  Returns 3×3 SE(2) transform (NaN if failed)."""
    T = np.full((3, 3), np.nan)
    n_p = p.shape[0]
    nl = ns.shape[0]

    if n_p < 2:
        return T

    # Step 1: initial rotation from points
    H0 = cross_covariance_2x2(p, q,
                               np.empty((0, 2)), np.empty((0, 2)), 1.0, 0.0)
    R0, s0 = svd_rotation_2x2(H0)
    if s0 < eps:
        return T

    # Step 2: sign-align normals
    if nl > 0:
        nt_a, ot_a = sign_align_normals(R0, ns, nt, ot)
    else:
        nt_a = np.empty((0, 2))
        ot_a = np.empty(0)

    # Step 3: final rotation with all points + aligned normals
    H1 = cross_covariance_2x2(p, q, ns, nt_a, W_P, W_D)
    R1, s1 = svd_rotation_2x2(H1)
    if s1 < eps:
        return T

    # Step 4: translation
    t = solve_translation_2x2(R1, p, q, ns, os_, ot_a, W_P, W_L_M)

    # Step 5: assemble inv(T)  —  T_inv = [[R^T, -R^T t], [0 0 1]]
    T[0, 0] = R1[0, 0]; T[0, 1] = R1[1, 0]
    T[1, 0] = R1[0, 1]; T[1, 1] = R1[1, 1]
    T[0, 2] = -(R1[0, 0] * t[0] + R1[1, 0] * t[1])
    T[1, 2] = -(R1[0, 1] * t[0] + R1[1, 1] * t[1])
    T[2, 0] = 0.0; T[2, 1] = 0.0; T[2, 2] = 1.0

    return T
