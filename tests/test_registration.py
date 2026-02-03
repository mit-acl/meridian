"""
Tests for registration with planes and gravity constraints.

These tests verify that the Registerer correctly recovers transformations
using various combinations of points, lines, planes, and gravity directions.
All tests provide true correspondences (no matching involved).

Tests are designed to fail until plane and gravity registration is implemented.
"""

import pytest
import numpy as np
import robotdatapy as rdp

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint, SegmentPlane
from gen_seg_match.params import RegisterParams
from gen_seg_match.register.registerer import Registerer


def assert_transforms_equal(T_est: np.ndarray, T_gt: np.ndarray, rel_tol=1e-6):
    """Helper to assert two 4x4 transforms are approximately equal."""
    for i in range(4):
        for j in range(4):
            assert pytest.approx(T_est[i, j], rel=rel_tol) == T_gt[i, j]


@pytest.fixture
def default_register_params():
    return RegisterParams()


@pytest.fixture
def sample_transform():
    """A sample 3D rigid transformation with rotation and translation."""
    return rdp.transform.xyz_rpy_to_transform(
        np.array([0.5, -0.3, 0.2]), np.array([0.1, 0.15, np.pi / 6])
    )


# =============================================================================
# Test 1: Point + Non-horizontal Line + Gravity
# =============================================================================
class TestPointLineGravity:
    """
    Test registration with a single point, a non-horizontal line, and gravity.

    DOF analysis:
    - Gravity direction: constrains 2 rotation DOF (pitch and roll)
    - Non-horizontal line: constrains 1 rotation DOF (yaw) + 2 translation DOF
    - Point: constrains remaining 1 translation DOF (along line direction)
    Total: 6 DOF fully constrained
    """

    def test_point_line_gravity_basic(self, default_register_params, sample_transform):
        """
        Basic test with a point, non-horizontal line, and gravity.
        The line is oriented at 45 degrees from horizontal.
        """
        # Source frame segments
        point_a = SegmentPoint(id=1, point=np.array([1.0, 2.0, 0.5]))
        line_a = SegmentLine(
            id=2,
            point=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 1.0]),  # 45 deg from horizontal
        )
        gravity_a = np.array([0.0, 0.0, -1.0])

        # Target frame: copy and transform
        point_b = point_a.copy()
        line_b = line_a.copy()
        point_b.transform(sample_transform)
        line_b.transform(sample_transform)
        gravity_b = sample_transform[:3, :3] @ gravity_a

        source = [point_a, line_a]
        target = [point_b, line_b]
        correspondences = np.array([[1, 1], [2, 2]])

        registerer = Registerer(default_register_params)
        result = registerer.register(
            source, target, gravity_a, gravity_b, correspondences=correspondences
        )

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)

    def test_point_line_gravity_flipped_direction(
        self, default_register_params, sample_transform
    ):
        """
        Test with flipped line direction in target to verify sign resolution.
        The registerer should handle the sign ambiguity correctly.
        """
        point_a = SegmentPoint(id=1, point=np.array([1.0, 2.0, 0.5]))
        line_a = SegmentLine(
            id=2,
            point=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 1.0, 1.0]),  # diagonal line
        )
        gravity_a = np.array([0.0, 0.0, -1.0])

        point_b = point_a.copy()
        line_b = line_a.copy()
        point_b.transform(sample_transform)
        line_b.transform(sample_transform)
        gravity_b = sample_transform[:3, :3] @ gravity_a

        # Flip the line direction in target
        line_b.direction = -line_b.direction

        source = [point_a, line_a]
        target = [point_b, line_b]
        correspondences = np.array([[1, 1], [2, 2]])

        registerer = Registerer(default_register_params)
        result = registerer.register(
            source, target, gravity_a, gravity_b, correspondences=correspondences
        )

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)


# # =============================================================================
# # Test 2: Plane + Line (No Gravity)
# # =============================================================================
# class TestPlaneLineNoGravity:
#     """
#     Test registration with a single plane and line, without gravity.

#     The plane and line must be non-orthogonal and non-parallel to fully
#     constrain the 6 DOF transformation.

#     DOF analysis:
#     - Plane normal: constrains 2 rotation DOF
#     - Line direction (non-parallel to normal): constrains 1 rotation DOF
#     - Plane distance: constrains 1 translation DOF
#     - Line position: constrains 2 translation DOF
#     Total: 6 DOF fully constrained
#     """

#     def test_plane_line_basic(self, default_register_params, sample_transform):
#         """
#         Basic test with a plane and line at ~45 degrees to each other.
#         """
#         # Plane with normal pointing mostly up but tilted
#         plane_a = SegmentPlane(
#             id=1,
#             point=np.array([0.0, 0.0, 1.0]),
#             normal=np.array([0.0, 0.3, 1.0]),  # tilted from vertical
#         )
#         # Line at an angle to the plane (not parallel, not perpendicular)
#         line_a = SegmentLine(
#             id=2,
#             point=np.array([1.0, 0.0, 0.0]),
#             direction=np.array([1.0, 1.0, 0.5]),
#         )

#         plane_b = plane_a.copy()
#         line_b = line_a.copy()
#         plane_b.transform(sample_transform)
#         line_b.transform(sample_transform)

#         source = [plane_a, line_a]
#         target = [plane_b, line_b]
#         correspondences = np.array([[1, 1], [2, 2]])

#         registerer = Registerer(default_register_params)
#         result = registerer.register(source, target, correspondences=correspondences)

#         T_gt = np.linalg.inv(sample_transform)
#         assert_transforms_equal(result.transformation, T_gt)

#     def test_plane_line_flipped_normal(self, default_register_params, sample_transform):
#         """
#         Test with flipped plane normal in target to verify sign resolution.
#         """
#         plane_a = SegmentPlane(
#             id=1,
#             point=np.array([0.0, 0.0, 1.0]),
#             normal=np.array([0.2, 0.3, 1.0]),
#         )
#         line_a = SegmentLine(
#             id=2,
#             point=np.array([1.0, 0.0, 0.0]),
#             direction=np.array([1.0, 1.0, 0.5]),
#         )

#         plane_b = plane_a.copy()
#         line_b = line_a.copy()
#         plane_b.transform(sample_transform)
#         line_b.transform(sample_transform)

#         # Flip the plane normal in target
#         plane_b.normal = -plane_b.normal

#         source = [plane_a, line_a]
#         target = [plane_b, line_b]
#         correspondences = np.array([[1, 1], [2, 2]])

#         registerer = Registerer(default_register_params)
#         result = registerer.register(source, target, correspondences=correspondences)

#         T_gt = np.linalg.inv(sample_transform)
#         assert_transforms_equal(result.transformation, T_gt)

#     def test_plane_line_different_point_representation(
#         self, default_register_params, sample_transform
#     ):
#         """
#         Test where target plane/line use different points on the same geometric entity.
#         The plane point is shifted along the plane, and line point is shifted along the line.
#         """
#         plane_a = SegmentPlane(
#             id=1,
#             point=np.array([0.0, 0.0, 1.0]),
#             normal=np.array([0.0, 0.0, 1.0]),  # horizontal plane at z=1
#         )
#         line_a = SegmentLine(
#             id=2,
#             point=np.array([0.0, 0.0, 0.0]),
#             direction=np.array([1.0, 1.0, 0.5]),
#         )

#         plane_b = plane_a.copy()
#         line_b = line_a.copy()
#         plane_b.transform(sample_transform)
#         line_b.transform(sample_transform)

#         # Shift plane_b's point to a different location on the same plane
#         # (move in the plane's tangent directions)
#         tangent1 = np.array([1.0, 0.0, 0.0])
#         tangent1_transformed = sample_transform[:3, :3] @ tangent1
#         plane_b.point = plane_b.point + 2.0 * tangent1_transformed

#         # Shift line_b's point along the line direction
#         line_b.point = line_b.point + 1.5 * line_b.direction

#         source = [plane_a, line_a]
#         target = [plane_b, line_b]
#         correspondences = np.array([[1, 1], [2, 2]])

#         registerer = Registerer(default_register_params)
#         result = registerer.register(source, target, correspondences=correspondences)

#         T_gt = np.linalg.inv(sample_transform)
#         assert_transforms_equal(result.transformation, T_gt)


# =============================================================================
# Test 3: Plane + Point + Gravity
# =============================================================================
class TestPlanePointGravity:
    """
    Test registration with a non-horizontal plane, point, and gravity.

    DOF analysis:
    - Gravity: constrains 2 rotation DOF (pitch and roll)
    - Non-horizontal plane normal: constrains 1 rotation DOF (yaw)
    - Plane distance: constrains 1 translation DOF
    - Point: constrains 2 more translation DOF
    Total: 6 DOF fully constrained
    """

    def test_plane_point_gravity_basic(self, default_register_params, sample_transform):
        """
        Basic test with a tilted plane, a point, and gravity.
        """
        # Non-horizontal plane (tilted ~30 deg from horizontal)
        plane_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.5, 0.0, 1.0]),
        )
        point_a = SegmentPoint(id=2, point=np.array([2.0, 1.0, 0.0]))
        gravity_a = np.array([0.0, 0.0, -1.0])

        plane_b = plane_a.copy()
        point_b = point_a.copy()
        plane_b.transform(sample_transform)
        point_b.transform(sample_transform)
        gravity_b = sample_transform[:3, :3] @ gravity_a

        source = [plane_a, point_a]
        target = [plane_b, point_b]
        correspondences = np.array([[1, 1], [2, 2]])

        registerer = Registerer(default_register_params)
        result = registerer.register(
            source, target, gravity_a, gravity_b, correspondences=correspondences
        )

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)

    def test_plane_point_gravity_flipped_normal(
        self, default_register_params, sample_transform
    ):
        """
        Test with flipped plane normal to verify sign resolution with gravity.
        """
        plane_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.3, 0.4, 1.0]),
        )
        point_a = SegmentPoint(id=2, point=np.array([2.0, 1.0, 0.0]))
        gravity_a = np.array([0.0, 0.0, -1.0])

        plane_b = plane_a.copy()
        point_b = point_a.copy()
        plane_b.transform(sample_transform)
        point_b.transform(sample_transform)
        gravity_b = sample_transform[:3, :3] @ gravity_a

        # Flip the plane normal
        plane_b.normal = -plane_b.normal

        source = [plane_a, point_a]
        target = [plane_b, point_b]
        correspondences = np.array([[1, 1], [2, 2]])

        registerer = Registerer(default_register_params)
        result = registerer.register(
            source, target, gravity_a, gravity_b, correspondences=correspondences
        )

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)


# =============================================================================
# Test 4: Two Planes + One Line
# =============================================================================
class TestTwoPlanesOneLine:
    """
    Test registration with two non-parallel planes and one line.

    DOF analysis:
    - Two non-parallel plane normals: constrain 3 rotation DOF
    - Two plane distances: constrain 2 translation DOF
    - Line (not parallel to plane intersection): constrains 1 translation DOF
    Total: 6 DOF fully constrained
    """

    def test_two_planes_one_line_basic(
        self, default_register_params, sample_transform
    ):
        """
        Two non-parallel planes and a line not parallel to their intersection.
        """
        plane1_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),  # horizontal
        )
        plane2_a = SegmentPlane(
            id=2,
            point=np.array([1.0, 0.0, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),  # vertical, facing +x
        )
        # Line not parallel to the y-axis (intersection of the two planes)
        line_a = SegmentLine(
            id=3,
            point=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 1.0, 1.0]),
        )

        plane1_b = plane1_a.copy()
        plane2_b = plane2_a.copy()
        line_b = line_a.copy()
        plane1_b.transform(sample_transform)
        plane2_b.transform(sample_transform)
        line_b.transform(sample_transform)

        source = [plane1_a, plane2_a, line_a]
        target = [plane1_b, plane2_b, line_b]
        correspondences = np.array([[1, 1], [2, 2], [3, 3]])

        registerer = Registerer(default_register_params)
        result = registerer.register(source, target, correspondences=correspondences)

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)


# =============================================================================
# Test 5: Three Non-Parallel Planes
# =============================================================================
class TestThreeNonParallelPlanes:
    """
    Test registration with three mutually non-parallel planes.

    DOF analysis:
    - Three non-parallel normals: constrain 3 rotation DOF
    - Three plane distances: constrain 3 translation DOF
    Total: 6 DOF fully constrained
    """

    def test_three_planes_basic(self, default_register_params, sample_transform):
        """
        Three mutually non-parallel planes (like corner of a room).
        """
        plane1_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 0.0]),
            normal=np.array([1.0, 1.0, 0.0]),  # yz-plane
        )
        plane2_a = SegmentPlane(
            id=2,
            point=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 1.0]),  # xz-plane
        )
        plane3_a = SegmentPlane(
            id=3,
            point=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),  # xy-plane
        )

        plane1_b = plane1_a.copy()
        plane2_b = plane2_a.copy()
        plane3_b = plane3_a.copy()
        plane1_b.transform(sample_transform)
        plane2_b.transform(sample_transform)
        plane3_b.transform(sample_transform)

        source = [plane1_a, plane2_a, plane3_a]
        target = [plane1_b, plane2_b, plane3_b]
        correspondences = np.array([[1, 1], [2, 2], [3, 3]])

        registerer = Registerer(default_register_params)
        result = registerer.register(source, target, correspondences=correspondences)

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)

    def test_three_planes_all_normals_flipped(
        self, default_register_params, sample_transform
    ):
        """
        Three planes with all normals flipped in target to stress-test sign resolution.
        """
        plane1_a = SegmentPlane(
            id=1,
            point=np.array([1.0, 0.0, 0.0]),
            normal=np.array([1.0, 1.0, 0.0]),
        )
        plane2_a = SegmentPlane(
            id=2,
            point=np.array([0.0, 2.0, 0.0]),
            normal=np.array([0.0, 1.0, 1.0]),
        )
        plane3_a = SegmentPlane(
            id=3,
            point=np.array([0.0, 0.0, 3.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        )

        plane1_b = plane1_a.copy()
        plane2_b = plane2_a.copy()
        plane3_b = plane3_a.copy()
        plane1_b.transform(sample_transform)
        plane2_b.transform(sample_transform)
        plane3_b.transform(sample_transform)

        # Flip all normals
        plane1_b.normal = -plane1_b.normal
        plane2_b.normal = -plane2_b.normal
        plane3_b.normal = -plane3_b.normal

        source = [plane1_a, plane2_a, plane3_a]
        target = [plane1_b, plane2_b, plane3_b]
        correspondences = np.array([[1, 1], [2, 2], [3, 3]])

        registerer = Registerer(default_register_params)
        result = registerer.register(source, target, correspondences=correspondences)

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)


# =============================================================================
# Test 6: Horizontal Plane + Two Points + Gravity
# =============================================================================
class TestHorizontalPlaneTwoPointsGravity:
    """
    Test registration with a horizontal plane, two points, and gravity.

    This tests the case where the plane normal is aligned with gravity,
    so the plane alone doesn't constrain yaw.

    DOF analysis:
    - Gravity: constrains 2 rotation DOF (pitch and roll)
    - Horizontal plane (parallel to gravity): constrains 0 additional rotation DOF
      but constrains 1 translation DOF (vertical)
    - Two points: constrain 1 rotation DOF (yaw) + 2 translation DOF (horizontal)
    Total: 6 DOF fully constrained
    """

    def test_horizontal_plane_two_points_gravity(
        self, default_register_params, sample_transform
    ):
        """
        Horizontal plane with two non-coincident points and gravity.
        """
        # Horizontal plane
        plane_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        )
        # Two points that aren't vertically aligned
        point1_a = SegmentPoint(id=2, point=np.array([1.0, 0.0, 0.0]))
        point2_a = SegmentPoint(id=3, point=np.array([0.0, 2.0, 0.5]))
        gravity_a = np.array([0.0, 0.0, -1.0])

        plane_b = plane_a.copy()
        point1_b = point1_a.copy()
        point2_b = point2_a.copy()
        plane_b.transform(sample_transform)
        point1_b.transform(sample_transform)
        point2_b.transform(sample_transform)
        gravity_b = sample_transform[:3, :3] @ gravity_a

        source = [plane_a, point1_a, point2_a]
        target = [plane_b, point1_b, point2_b]
        correspondences = np.array([[1, 1], [2, 2], [3, 3]])

        registerer = Registerer(default_register_params)
        result = registerer.register(
            source, target, gravity_a, gravity_b, correspondences=correspondences
        )

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)


# =============================================================================
# Test 7: Mixed Segments - Plane + Line + Point
# =============================================================================
class TestMixedSegments:
    """
    Test registration with a mix of plane, line, and point.

    DOF analysis:
    - Plane normal: 2 rotation DOF
    - Line direction (non-parallel to normal): 1 rotation DOF
    - Plane distance: 1 translation DOF
    - Line position: 1 translation DOF (the one not covered by plane)
    - Point: 1 translation DOF (the remaining one)
    Total: 6 DOF fully constrained
    """

    def test_plane_line_point_basic(self, default_register_params, sample_transform):
        """
        A plane, a line not parallel to the plane, and a point.
        """
        plane_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 2.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        )
        line_a = SegmentLine(
            id=2,
            point=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 1.0]),  # 45 deg to horizontal
        )
        point_a = SegmentPoint(id=3, point=np.array([1.0, 1.0, 1.0]))

        plane_b = plane_a.copy()
        line_b = line_a.copy()
        point_b = point_a.copy()
        plane_b.transform(sample_transform)
        line_b.transform(sample_transform)
        point_b.transform(sample_transform)

        source = [plane_a, line_a, point_a]
        target = [plane_b, line_b, point_b]
        correspondences = np.array([[1, 1], [2, 2], [3, 3]])

        registerer = Registerer(default_register_params)
        result = registerer.register(source, target, correspondences=correspondences)

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)

    def test_plane_line_point_all_flipped(
        self, default_register_params, sample_transform
    ):
        """
        Same as above but with both plane normal and line direction flipped.
        """
        plane_a = SegmentPlane(
            id=1,
            point=np.array([0.0, 0.0, 2.0]),
            normal=np.array([0.2, 0.3, 1.0]),
        )
        line_a = SegmentLine(
            id=2,
            point=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.5, 1.0]),
        )
        point_a = SegmentPoint(id=3, point=np.array([1.0, 1.0, 1.0]))

        plane_b = plane_a.copy()
        line_b = line_a.copy()
        point_b = point_a.copy()
        plane_b.transform(sample_transform)
        line_b.transform(sample_transform)
        point_b.transform(sample_transform)

        # Flip both
        plane_b.normal = -plane_b.normal
        line_b.direction = -line_b.direction

        source = [plane_a, line_a, point_a]
        target = [plane_b, line_b, point_b]
        correspondences = np.array([[1, 1], [2, 2], [3, 3]])

        registerer = Registerer(default_register_params)
        result = registerer.register(source, target, correspondences=correspondences)

        T_gt = np.linalg.inv(sample_transform)
        assert_transforms_equal(result.transformation, T_gt)
