import pytest
import numpy as np
import robotdatapy.transform as rdpt

from gen_seg_match.segment.segment_types import SegmentLine
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.params.segment_match_params import SegmentMatchParams


@pytest.fixture
def three_line_segments():
    linea1 = SegmentLine(
        id=1,
        point=np.array([0.0, 0.0, 0.0]),
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
            rdpt.xyz_rpy_to_transform(
                np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, np.pi / 6])
            )
        )

    return linesa, linesb


@pytest.fixture
def default_matcher_params():
    return SegmentMatchParams(
        dim=3,
        ratio_feature_dim=0,
        cos_feature_dim=0,
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


def test_line_distance_1(default_matcher_params, three_line_segments):
    linesa, linesb = three_line_segments
    matcher = SegmentMatcher(default_matcher_params)
    M, C, A = matcher.get_MCA(linesa, linesb)
    assert M.shape == (9, 9)

    # correct matches are (0a, 0b), (1a, 1b), (2a, 2b)
    M_0a0b_1a1b = M[0, 4]
    M_0a0b_2a2b = M[0, 8]
    M_1a1b_2a2b = M[4, 8]
    assert pytest.approx(M_0a0b_1a1b) == 1.0
    assert pytest.approx(M_0a0b_2a2b) == 1.0
    assert pytest.approx(M_1a1b_2a2b) == 1.0

    M_0a0b_1a2b = M[0, 5]
    M_0a0b_1a2b_expected_dist_diff = np.exp(
        -(1**2) / (2 * matcher.params.sigma_dist**2)
    )
    M_0a0b_1a2b_expected_angle_diff = np.exp(
        -(np.deg2rad(45) ** 2) / (2 * matcher.params.sigma_angle_rad**2)
    )
    M_0a0b_1a2b_expected = np.sqrt(
        M_0a0b_1a2b_expected_dist_diff * M_0a0b_1a2b_expected_angle_diff
    )
    print(M_0a0b_1a2b, M_0a0b_1a2b_expected)
    assert pytest.approx(M_0a0b_1a2b) == M_0a0b_1a2b_expected

    M_0a0b_2a1b = M[0, 7]
    M_0a0b_2a1b_expected = M_0a0b_1a2b_expected  # symmetric case
    assert pytest.approx(M_0a0b_2a1b) == M_0a0b_2a1b_expected


def test_line_distance_with_semantics_1(default_matcher_params, three_line_segments):
    linesa, linesb = three_line_segments
    params = default_matcher_params
    params.cos_feature_dim = 6
    matcher = SegmentMatcher(params)
    M, C, A = matcher.get_MCA(linesa, linesa)
    assert M.shape == (9, 9)
    # correct matches are (0a, 0a), (1a, 1a), (2a, 2a)
    M_0a0a_1a1a = M[0, 4]
    M_0a0a_2a2a = M[0, 8]
    M_1a1a_2a2a = M[4, 8]
    assert pytest.approx(M_0a0a_1a1a) == 1.0
    assert pytest.approx(M_0a0a_2a2a) == 1.0
    assert pytest.approx(M_1a1a_2a2a) == 1.0

    M, C, A = matcher.get_MCA(linesa, linesb)
    # correct matches are (0a, 0b), (1a, 1b), (2a, 2b)
    M_0a0b_1a1b = M[0, 4]
    M_0a0b_2a2b = M[0, 8]
    M_1a1b_2a2b = M[4, 8]
    assert np.allclose(linesa[0].cos_feature, linesb[0].cos_feature)
    assert np.allclose(linesa[1].cos_feature, linesb[1].cos_feature)
    assert np.allclose(linesa[2].cos_feature, linesb[2].cos_feature)
    assert pytest.approx(M_0a0b_1a1b) == 1.0
    assert pytest.approx(M_0a0b_2a2b) == 1.0
    assert pytest.approx(M_1a1b_2a2b) == 1.0

    M_0a0b_1a2b = M[0, 5]
    M_0a0b_1a2b_expected_dist_diff = np.exp(
        -(1**2) / (2 * matcher.params.sigma_dist**2)
    )
    M_0a0b_1a2b_expected_angle_diff = np.exp(
        -(np.deg2rad(45) ** 2) / (2 * matcher.params.sigma_angle_rad**2)
    )
    M_0a0b_1a2b_expected_pairwise = np.sqrt(
        M_0a0b_1a2b_expected_dist_diff * M_0a0b_1a2b_expected_angle_diff
    )
    M_0a0b_1a2b_expected_cosine1 = np.dot(
        linesa[0].cos_feature, linesb[0].cos_feature
    ) / (np.linalg.norm(linesa[0].cos_feature) * np.linalg.norm(linesb[0].cos_feature))
    M_0a0b_1a2b_expected_cosine2 = np.dot(
        linesa[1].cos_feature, linesb[2].cos_feature
    ) / (np.linalg.norm(linesa[1].cos_feature) * np.linalg.norm(linesb[2].cos_feature))
    M_0a0b_1a2b_expected = (
        M_0a0b_1a2b_expected_pairwise
        * M_0a0b_1a2b_expected_cosine1
        * M_0a0b_1a2b_expected_cosine2
    ) ** (1 / 3)
    print(M_0a0b_1a2b, M_0a0b_1a2b_expected)
    assert pytest.approx(M_0a0b_1a2b) == M_0a0b_1a2b_expected

    M_0a0b_2a1b = M[0, 7]
    M_0a0b_2a1b_expected = M_0a0b_1a2b_expected  # symmetric case
    assert pytest.approx(M_0a0b_2a1b) == M_0a0b_2a1b_expected
