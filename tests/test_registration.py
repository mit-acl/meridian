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

from meridian.primitive.primitive import LinePrimitive, PointPrimitive
from meridian.params import RegisterParams
from meridian.register.r3d.registerer import Registerer


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
        point_a = PointPrimitive(id=1, point=np.array([1.0, 2.0, 0.5]))
        line_a = LinePrimitive(
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
        point_a = PointPrimitive(id=1, point=np.array([1.0, 2.0, 0.5]))
        line_a = LinePrimitive(
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

