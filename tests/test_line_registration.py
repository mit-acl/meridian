import pytest
import numpy as np
import robotdatapy as rdp

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.params.segment_match_params import SegmentMatchParams
from gen_seg_match.params import RegisterParams
from gen_seg_match.register.registerer import Registerer


@pytest.fixture
def three_line_segments():
    linea1 = SegmentLine(
        id=1,
        point=np.array([-1.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        cos_feature=np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    )

    linea2 = SegmentLine(
        id=2,
        point=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 1.0]),
        cos_feature=np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0]),
    )

    linea3 = SegmentLine(
        id=3,
        point=np.array([2.0, 0.0, 1.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        cos_feature=np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]),
    )

    lineb1 = linea1.copy()
    lineb2 = linea2.copy()
    lineb3 = linea3.copy()

    lineb1.endpoints = (np.array([-1.0, 0.0, 0.0]), None)
    lineb2.endpoints = (np.array([-1.0, 0.0, -1.0]), np.array([2.0, 0.0, 2.0]))

    linesa = [linea1, linea2, linea3]
    linesb = [lineb1, lineb2, lineb3]

    for line in linesb:
        line.transform(
            rdp.transform.xyz_rpy_to_transform(
                np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi / 6])
            )
        )

    return linesa, linesb


@pytest.fixture
def default_matcher_params():
    return SegmentMatchParams(
        dim=3,
        ratio_feature_dim=0,
        cos_feature_dim=6,
        sigma_dist=1.0,
        epsilon_dist=1.0,
        min_dist=0.0,
        sigma_angle_rad=np.deg2rad(90.0),
        epsilon_angle_rad=np.deg2rad(90.0),
        min_angle_rad=0.0,
        distance_weight=1.0,
        ratio_weight=1.0,
        cosine_weight=1.0,
        ratio_epsilon=np.zeros(0),
        cosine_min=0.0,
        cosine_max=1.0,
        gravity_guided=False,
        gravity_unc_ang_rad=0.0,
    )


@pytest.fixture
def default_register_params():
    return RegisterParams()


def test_point_registration_1(default_register_params, three_line_segments):
    """Test registration with points only"""
    linesa, linesb = three_line_segments
    pointsa = [SegmentPoint(id=line.id, point=line.point) for line in linesa]
    pointsb = [SegmentPoint(id=line.id, point=line.point) for line in linesb]
    registerer = Registerer(default_register_params)
    correspondences = np.array([[1, 1], [2, 2], [3, 3]])
    T_pointsa_pointsb_est = registerer.register(
        pointsa, pointsb, correspondences=correspondences
    ).transformation
    T_pointsb_pointsa_gt = rdp.transform.xyz_rpy_to_transform(
        np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi / 6])
    )
    T_pointsa_pointsb_gt = np.linalg.inv(T_pointsb_pointsa_gt)
    for i in range(4):
        for j in range(4):
            assert (
                pytest.approx(T_pointsa_pointsb_est[i, j]) == T_pointsa_pointsb_gt[i, j]
            )


def test_line_registration_1(default_register_params, three_line_segments):
    """Test registration with KNOWN correspondences"""
    linesa, linesb = three_line_segments
    registerer = Registerer(default_register_params)
    correspondences = np.array([[1, 1], [2, 2], [3, 3]])
    T_linesa_linesb_est = registerer.register(
        linesa, linesb, correspondences=correspondences
    ).transformation
    T_linesb_linesa_gt = rdp.transform.xyz_rpy_to_transform(
        np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi / 6])
    )
    T_linesa_linesb_gt = np.linalg.inv(T_linesb_linesa_gt)
    for i in range(4):
        for j in range(4):
            assert pytest.approx(T_linesa_linesb_est[i, j]) == T_linesa_linesb_gt[i, j]


def test_line_registration_2(
    default_matcher_params, default_register_params, three_line_segments
):
    """Test registration with UNKNOWN correspondences"""
    linesa, linesb = three_line_segments
    matcher = SegmentMatcher(default_matcher_params)
    registerer = Registerer(default_register_params)
    correspondences = matcher.match(linesa, linesb)
    T_linesa_linesb_est = registerer.register(
        linesa, linesb, correspondences=correspondences
    ).transformation
    T_linesb_linesa_gt = rdp.transform.xyz_rpy_to_transform(
        np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi / 6])
    )
    T_linesa_linesb_gt = np.linalg.inv(T_linesb_linesa_gt)
    for i in range(4):
        for j in range(4):
            assert pytest.approx(T_linesa_linesb_est[i, j]) == T_linesa_linesb_gt[i, j]
