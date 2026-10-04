# TEMP(fitness-simplify): tests for the parameter-reduction study switches; delete with them.
import numpy as np
import pytest

from meridian.cross_view.matching import CrossViewMatching
from meridian.params.primitive_match_params import PrimitiveMatchParams
from meridian.params.register_params import RegisterParams
from meridian.primitive.primitive import PointPrimitive
from meridian.primitive.primitive_list import PrimitiveList
from tests.test_alignment_fitness import box_submap, compute


def points(xs):
    return PrimitiveList(
        [PointPrimitive(id=i, point=np.array([float(x), 0.0])) for i, x in enumerate(xs)]
    )


def three_of_four():
    """3 of 4 ground points land on aerial points."""
    patch = box_submap(ids_start=100, extent=40.0)
    return patch + points([1, 2, 3]), points([1, 2, 3, 30.5])


class TestWilsonSwitch:
    def test_off_scores_the_plain_inlier_ratio(self):
        aerial, ground = three_of_four()
        off = compute(aerial, ground, min_in_patch=0, min_inliers=0, use_wilson=False)
        on = compute(aerial, ground, min_in_patch=0, min_inliers=0)
        assert off.fitness == off.inlier_ratio == pytest.approx(0.75)
        assert on.fitness < off.fitness

    def test_off_still_applies_min_inliers(self):
        aerial, ground = three_of_four()
        assert compute(aerial, ground, min_in_patch=0, min_inliers=4, use_wilson=False).fitness == 0.0
        assert compute(aerial, ground, min_in_patch=0, min_inliers=3, use_wilson=False).fitness == pytest.approx(0.75)


class TestThresholdSource:
    def setup_method(self):
        self.mp = PrimitiveMatchParams(epsilon_dist=1.7, epsilon_angle_rad=np.deg2rad(12.0))
        self.rp = RegisterParams(cluster_trans_thresh_m=0.8, cluster_rot_thresh_deg=4.0)

    def resolve(self, source):
        self.rp.fitness_thresholds_from = source
        return CrossViewMatching.fitness_thresholds(self.mp, self.rp)

    def test_shipped_reads_the_fitness_fields(self):
        assert self.resolve("shipped") == pytest.approx((1.0, 2.0, np.deg2rad(5.0)))

    def test_matcher_ties_both_distances_to_epsilon_dist(self):
        assert self.resolve("matcher") == pytest.approx((1.7, 1.7, np.deg2rad(12.0)))

    def test_cluster_ties_both_distances_to_cluster_trans(self):
        assert self.resolve("cluster") == pytest.approx((0.8, 0.8, np.deg2rad(4.0)))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            self.resolve("nope")
