"""
Tests for translation-only point matching in 2D and 3D.

Verifies that SegmentMatcher with constrained invariants
(xy_dir_constrained_2d for 2D, xyz_dir_constrained for 3D) produces
matches whose per-axis within-map differences are consistent — the
geometric property guaranteed by a pure-translation transformation.

The consistency check mirrors the invariant implemented in
clipper/src/invariants/general_segment_distance.cpp:
    For matched pairs (ai<->bi, aj<->bj):
        2D: |((ai[k] - aj[k]) - (bi[k] - bj[k]))| <= sqrt(1/2) * epsilon  for k in {x,y}
        3D: |((ai[k] - aj[k]) - (bi[k] - bj[k]))| <= sqrt(1/3) * epsilon  for k in {x,y,z}
"""

import pytest
import numpy as np

from meridian.segment.segment_types import SegmentPoint, SegmentList
from meridian.match.segment_matcher import SegmentMatcher
from meridian.params.segment_match_params import SegmentMatchParams

SQRT_ONE_HALF = 0.70710678118
SQRT_ONE_THIRD = 0.57735026919


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_point_list(positions, start_id=0):
    """Create a SegmentList of SegmentPoints from a (N, dim) positions array."""
    return SegmentList(
        [
            SegmentPoint(id=start_id + i, point=np.array(pos, dtype=float))
            for i, pos in enumerate(positions)
        ]
    )


def check_pairwise_consistency(map1, map2, matches, epsilon_dist, dim):
    """
    Verify that every pair of returned matches satisfies the per-axis
    constrained invariant.  Points must be given in the *aligned* frame
    (i.e. after the matcher's internal frame alignment has been applied).
    """
    if len(matches) < 2:
        return

    scale = SQRT_ONE_HALF if dim == 2 else SQRT_ONE_THIRD
    threshold = scale * epsilon_dist

    map1_sl = map1 if isinstance(map1, SegmentList) else SegmentList(map1)
    map2_sl = map2 if isinstance(map2, SegmentList) else SegmentList(map2)

    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            ai = map1_sl.get_segment_from_id(int(matches[i][0])).point
            bi = map2_sl.get_segment_from_id(int(matches[i][1])).point
            aj = map1_sl.get_segment_from_id(int(matches[j][0])).point
            bj = map2_sl.get_segment_from_id(int(matches[j][1])).point

            for k in range(dim):
                c = abs((ai[k] - aj[k]) - (bi[k] - bj[k]))
                assert c <= threshold + 1e-9, (
                    f"Match pair ({matches[i].tolist()}, {matches[j].tolist()}) "
                    f"violates axis-{k} constraint: "
                    f"c={c:.4f} > threshold={threshold:.4f}"
                )


def rot2d(theta):
    """2D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotz(theta):
    """3D rotation matrix about the z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ===========================================================================
# 2D Translation-Only Matching  (xy_dir_constrained_2d)
# ===========================================================================


class TestTranslationOnly2D:
    @pytest.fixture
    def params(self):
        return SegmentMatchParams(
            dim=2,
            xy_dir_constrained_2d=True,
            sigma_dist=1.0,
            epsilon_dist=1.0,
            solver="clipper",
            cos_feature_dim=0,
            min_dist=0.0,
            rot_unc_ang_rad=0.0,
            point_noise_from_angle=False,
        )

    @pytest.fixture
    def axis_dirs(self):
        return dict(
            global_x_dir1=np.array([1.0, 0.0]),
            global_y_dir1=np.array([0.0, 1.0]),
            global_x_dir2=np.array([1.0, 0.0]),
            global_y_dir2=np.array([0.0, 1.0]),
        )

    # ---- known-match tests ------------------------------------------------

    def test_known_matches_pure_translation(self, params, axis_dirs):
        """Pure translation, both maps axis-aligned → all matches found."""
        positions = [[0, 0], [5, 0], [0, 5], [3, 4], [7, 2]]
        t = np.array([2.0, 3.0])

        map1 = make_point_list(positions)
        map2 = make_point_list([np.array(p) + t for p in positions])

        matches = (
            SegmentMatcher(params).match(map1, map2, **axis_dirs).association_array
        )

        assert len(matches) == 5
        for m in matches:
            assert m[0] == m[1]

    def test_known_matches_with_rotation(self, params):
        """Map2 rotated + translated; rotation supplied to matcher."""
        positions = [[0, 0], [5, 0], [0, 5], [3, 4], [7, 2]]
        R = rot2d(np.pi / 4)
        t = np.array([2.0, 3.0])

        map1 = make_point_list(positions)
        map2 = make_point_list([R @ np.array(p) + t for p in positions])

        matches = (
            SegmentMatcher(params)
            .match(
                map1,
                map2,
                global_x_dir1=np.array([1.0, 0.0]),
                global_y_dir1=np.array([0.0, 1.0]),
                global_x_dir2=R[:, 0],
                global_y_dir2=R[:, 1],
            )
            .association_array
        )

        assert len(matches) == 5
        for m in matches:
            assert m[0] == m[1]

    # ---- random-points consistency tests ----------------------------------

    def test_random_points_consistency(self, params, axis_dirs):
        """
        Random 2D points with no true correspondence.
        Whatever matches CLIPPER returns must satisfy the invariant.
        """
        rng = np.random.default_rng(42)
        map1 = make_point_list(rng.uniform(-20, 20, (15, 2)))
        map2 = make_point_list(rng.uniform(-20, 20, (15, 2)))

        matches = (
            SegmentMatcher(params).match(map1, map2, **axis_dirs).association_array
        )

        check_pairwise_consistency(map1, map2, matches, params.epsilon_dist, dim=2)

    def test_random_points_with_rotation_consistency(self, params):
        """
        Random 2D points, map2 rotated. Check invariant in aligned frame.
        """
        rng = np.random.default_rng(123)
        R = rot2d(np.pi / 6)

        map1_pos = rng.uniform(-20, 20, (15, 2))
        map2_pos_unrotated = rng.uniform(-20, 20, (15, 2))
        map2_pos = (R @ map2_pos_unrotated.T).T

        map1 = make_point_list(map1_pos)
        map2 = make_point_list(map2_pos)

        matches = (
            SegmentMatcher(params)
            .match(
                map1,
                map2,
                global_x_dir1=np.array([1.0, 0.0]),
                global_y_dir1=np.array([0.0, 1.0]),
                global_x_dir2=R[:, 0],
                global_y_dir2=R[:, 1],
            )
            .association_array
        )

        # Aligned frame: map1 unchanged, map2 → R^T @ map2_pos = unrotated
        map1_aligned = make_point_list(map1_pos)
        map2_aligned = make_point_list(map2_pos_unrotated)
        check_pairwise_consistency(
            map1_aligned,
            map2_aligned,
            matches,
            params.epsilon_dist,
            dim=2,
        )

    # ---- pathological-case test -------------------------------------------

    def test_rejects_inconsistent_distances(self, params, axis_dirs):
        """
        Two points 0.5 m apart in map1 vs 20 m apart in map2.
        Both cannot be matched simultaneously because the x-inconsistency
        |(0.5) - (20)| = 19.5 vastly exceeds epsilon.
        """
        map1 = make_point_list([[0.0, 0.0], [0.5, 0.0]])
        map2 = make_point_list([[0.0, 0.0], [20.0, 0.0]])

        matches = (
            SegmentMatcher(params).match(map1, map2, **axis_dirs).association_array
        )

        # At most 1 match; if 2 are returned the invariant check will fail.
        if len(matches) >= 2:
            check_pairwise_consistency(
                map1,
                map2,
                matches,
                params.epsilon_dist,
                dim=2,
            )


# ===========================================================================
# 3D Translation-Only Matching  (xyz_dir_constrained)
# ===========================================================================


class TestTranslationOnly3D:
    @pytest.fixture
    def params(self):
        return SegmentMatchParams(
            dim=3,
            xyz_dir_constrained=True,
            sigma_dist=1.0,
            epsilon_dist=1.0,
            solver="clipper",
            cos_feature_dim=0,
            min_dist=0.0,
            rot_unc_ang_rad=0.0,
            point_noise_from_angle=False,
        )

    @pytest.fixture
    def axis_dirs(self):
        return dict(
            global_x_dir1=np.array([1.0, 0.0, 0.0]),
            global_y_dir1=np.array([0.0, 1.0, 0.0]),
            global_z_dir1=np.array([0.0, 0.0, 1.0]),
            global_x_dir2=np.array([1.0, 0.0, 0.0]),
            global_y_dir2=np.array([0.0, 1.0, 0.0]),
            global_z_dir2=np.array([0.0, 0.0, 1.0]),
        )

    # ---- known-match tests ------------------------------------------------

    def test_known_matches_pure_translation(self, params, axis_dirs):
        """Pure 3D translation → all matches found."""
        positions = [[0, 0, 0], [5, 0, 1], [0, 5, 2], [3, 4, 1.5], [7, 2, 0.5]]
        t = np.array([2.0, 3.0, 1.0])

        map1 = make_point_list(positions)
        map2 = make_point_list([np.array(p) + t for p in positions])

        matches = (
            SegmentMatcher(params).match(map1, map2, **axis_dirs).association_array
        )

        assert len(matches) == 5
        for m in matches:
            assert m[0] == m[1]

    def test_known_matches_with_rotation(self, params):
        """Map2 rotated about z + translated; rotation supplied."""
        positions = [[0, 0, 0], [5, 0, 1], [0, 5, 2], [3, 4, 1.5], [7, 2, 0.5]]
        R = rotz(np.pi / 6)
        t = np.array([2.0, 3.0, 1.0])

        map1 = make_point_list(positions)
        map2 = make_point_list([R @ np.array(p) + t for p in positions])

        matches = (
            SegmentMatcher(params)
            .match(
                map1,
                map2,
                global_x_dir1=np.array([1.0, 0.0, 0.0]),
                global_y_dir1=np.array([0.0, 1.0, 0.0]),
                global_z_dir1=np.array([0.0, 0.0, 1.0]),
                global_x_dir2=R[:, 0],
                global_y_dir2=R[:, 1],
                global_z_dir2=R[:, 2],
            )
            .association_array
        )

        assert len(matches) == 5
        for m in matches:
            assert m[0] == m[1]

    # ---- random-points consistency tests ----------------------------------

    def test_random_points_consistency(self, params, axis_dirs):
        """
        Random 3D points with no true correspondence.
        Check xyz_dir_constrained invariant on returned matches.
        """
        rng = np.random.default_rng(42)
        map1 = make_point_list(rng.uniform(-20, 20, (15, 3)))
        map2 = make_point_list(rng.uniform(-20, 20, (15, 3)))

        matches = (
            SegmentMatcher(params).match(map1, map2, **axis_dirs).association_array
        )

        check_pairwise_consistency(map1, map2, matches, params.epsilon_dist, dim=3)

    def test_random_points_with_rotation_consistency(self, params):
        """
        Random 3D points, map2 rotated about z. Check invariant in aligned frame.
        """
        rng = np.random.default_rng(123)
        R = rotz(np.pi / 6)

        map1_pos = rng.uniform(-20, 20, (15, 3))
        map2_pos_unrotated = rng.uniform(-20, 20, (15, 3))
        map2_pos = (R @ map2_pos_unrotated.T).T

        map1 = make_point_list(map1_pos)
        map2 = make_point_list(map2_pos)

        matches = (
            SegmentMatcher(params)
            .match(
                map1,
                map2,
                global_x_dir1=np.array([1.0, 0.0, 0.0]),
                global_y_dir1=np.array([0.0, 1.0, 0.0]),
                global_z_dir1=np.array([0.0, 0.0, 1.0]),
                global_x_dir2=R[:, 0],
                global_y_dir2=R[:, 1],
                global_z_dir2=R[:, 2],
            )
            .association_array
        )

        # Aligned frame: map1 unchanged, map2 → R^T @ map2_pos = unrotated
        map1_aligned = make_point_list(map1_pos)
        map2_aligned = make_point_list(map2_pos_unrotated)
        check_pairwise_consistency(
            map1_aligned,
            map2_aligned,
            matches,
            params.epsilon_dist,
            dim=3,
        )

    # ---- pathological-case test -------------------------------------------

    def test_rejects_inconsistent_distances(self, params, axis_dirs):
        """
        Two points 0.5 m apart in map1 vs 20 m apart in map2.
        Simultaneous matching would violate the invariant.
        """
        map1 = make_point_list([[0, 0, 0], [0.5, 0, 0]])
        map2 = make_point_list([[0, 0, 0], [20, 0, 0]])

        matches = (
            SegmentMatcher(params).match(map1, map2, **axis_dirs).association_array
        )

        if len(matches) >= 2:
            check_pairwise_consistency(
                map1,
                map2,
                matches,
                params.epsilon_dist,
                dim=3,
            )
