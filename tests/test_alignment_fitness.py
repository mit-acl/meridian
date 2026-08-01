import time

import numpy as np
import pytest

from meridian.map3d.similarity_metrics import AlignmentFitness
from meridian.primitive.primitive import LinePrimitive, PointPrimitive
from meridian.primitive.primitive_list import PrimitiveList

PARAMS = dict(
    point_inlier_thresh_m=1.0,
    line_inlier_thresh_m=1.5,
    line_angle_thresh_rad=np.deg2rad(5.0),
    line_min_overlap=0.25,
    wilson_z=2.576,
    min_in_patch=10,
)

# Submap size that reproduces the runtime seen on real short-map-matching data.
REAL_SIZE = 110


def compute(aerial, ground, T=np.eye(4), **overrides):
    params = dict(PARAMS)
    params.update(overrides)
    return AlignmentFitness.compute(
        aerial_segments=aerial, ground_segments=ground, T_aerial_ground=T, **params
    )


def line(id, point, direction, endpoints=(None, None)):
    return LinePrimitive(
        id=id,
        point=np.asarray(point, dtype=float),
        direction=np.asarray(direction, dtype=float),
        endpoints=endpoints,
    )


def segment(id, p0, p1):
    return LinePrimitive.from_endpoints(
        id, np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    )


def ray(id, origin, direction):
    origin = np.asarray(origin, dtype=float)
    return line(id, origin, direction, endpoints=(origin, None))


def make_submaps(n_pts, n_lines, extent=60.0, seed=0, frac_endpoints=0.85):
    """A realistic aerial submap over ``[0, extent]^2`` plus a ground submap that
    is a noisy, partially overlapping, slightly rotated copy of it, and the
    transform that brings the ground submap back onto the aerial one.

    Lines are a street-grid-like mix of near-axis-aligned and oblique geometry,
    with a minority left as rays or infinite lines, so a realistic fraction of
    pairs passes the angle gate.
    """
    rng = np.random.default_rng(seed)
    aerial = PrimitiveList()
    nid = 0

    for p in rng.uniform(0, extent, size=(n_pts, 2)):
        aerial.append(PointPrimitive(id=nid, point=p.copy()))
        nid += 1

    for i in range(n_lines):
        p0 = rng.uniform(0, extent, size=2)
        base = 0.0 if i % 2 else np.pi / 2
        ang = base + rng.normal(0.0, np.deg2rad(12.0))
        d = np.array([np.cos(ang), np.sin(ang)])
        length = rng.uniform(3.0, 25.0)
        if rng.random() < frac_endpoints:
            aerial.append(segment(nid, p0, p0 + d * length))
        elif rng.random() < 0.5:
            aerial.append(ray(nid, p0, d))
        else:
            aerial.append(line(nid, p0, d))
        nid += 1

    theta = np.deg2rad(3.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([1.2, -0.8])
    to_ground = lambda p: R.T @ (p + rng.normal(0, 0.4, size=2) - t)  # noqa: E731

    ground = PrimitiveList()
    for keep, seg in zip(rng.random(len(aerial)) < 0.8, aerial):
        if not keep:
            continue
        gid = len(ground)
        if isinstance(seg, PointPrimitive):
            ground.append(PointPrimitive(id=gid, point=to_ground(seg.point)))
            continue
        ep0, ep1 = seg.endpoints
        if ep0 is not None and ep1 is not None:
            ground.append(segment(gid, to_ground(ep0), to_ground(ep1)))
        else:
            p = to_ground(seg.point)
            d = R.T @ seg.get_direction()
            eps = (p.copy(), None) if seg.num_endpoints == 1 else (None, None)
            ground.append(line(gid, p, d, endpoints=eps))

    T = np.eye(4)
    T[:2, :2] = R
    T[:2, 3] = t
    return aerial, ground, T


@pytest.fixture(scope="module")
def real_size_submaps():
    return make_submaps(REAL_SIZE, REAL_SIZE)


def box_submap(ids_start=0, extent=20.0):
    """Aerial primitives spanning a square patch, so the patch box is well
    conditioned regardless of what else a test adds."""
    return PrimitiveList(
        [
            PointPrimitive(id=ids_start, point=np.array([0.0, 0.0])),
            PointPrimitive(id=ids_start + 1, point=np.array([extent, extent])),
        ]
    )


class TestPoints:
    def test_inside_threshold_counts_as_inlier(self):
        aerial = box_submap()
        aerial.append(PointPrimitive(id=2, point=np.array([10.0, 10.0])))
        ground = PrimitiveList([PointPrimitive(id=0, point=np.array([10.5, 10.0]))])
        result = compute(aerial, ground, min_in_patch=1)
        assert result.n_in_patch == 1
        assert result.inlier_pairs == [(0, 2)]
        assert result.mean_inlier_quality == pytest.approx(0.5)

    def test_outside_threshold_counts_as_outlier(self):
        aerial = box_submap()
        aerial.append(PointPrimitive(id=2, point=np.array([10.0, 10.0])))
        ground = PrimitiveList([PointPrimitive(id=0, point=np.array([12.0, 10.0]))])
        result = compute(aerial, ground, min_in_patch=1)
        assert (result.n_in_patch, result.n_inliers) == (1, 0)

    def test_outside_patch_is_not_counted(self):
        aerial = box_submap()
        aerial.append(PointPrimitive(id=2, point=np.array([10.0, 10.0])))
        ground = PrimitiveList([PointPrimitive(id=0, point=np.array([100.0, 100.0]))])
        result = compute(aerial, ground, min_in_patch=1)
        assert (result.n_in_patch, result.n_inliers) == (0, 0)

    def test_nearest_aerial_point_wins(self):
        aerial = box_submap()
        aerial.append(PointPrimitive(id=2, point=np.array([10.0, 10.0])))
        aerial.append(PointPrimitive(id=3, point=np.array([10.4, 10.0])))
        ground = PrimitiveList([PointPrimitive(id=0, point=np.array([10.5, 10.0]))])
        assert compute(aerial, ground, min_in_patch=1).inlier_pairs == [(0, 3)]


class TestLines:
    def test_parallel_and_close_is_an_inlier(self):
        aerial = box_submap() + PrimitiveList([segment(2, [2.0, 10.0], [18.0, 10.0])])
        ground = PrimitiveList([segment(0, [4.0, 10.5], [16.0, 10.5])])
        result = compute(aerial, ground, min_in_patch=1)
        assert result.inlier_pairs == [(0, 2)]

    def test_angle_gate_rejects(self):
        aerial = box_submap() + PrimitiveList([segment(2, [2.0, 10.0], [18.0, 10.0])])
        d = np.array([np.cos(np.deg2rad(20.0)), np.sin(np.deg2rad(20.0))])
        ground = PrimitiveList([segment(0, [10.0, 10.2], [10.0, 10.2] + d * 8.0)])
        result = compute(aerial, ground, min_in_patch=1)
        assert (result.n_in_patch, result.n_inliers) == (1, 0)

    def test_reversed_direction_is_still_parallel(self):
        """A line's direction sign is arbitrary, so the angle must be folded into
        [0, pi/2]."""
        aerial = box_submap() + PrimitiveList([segment(2, [2.0, 10.0], [18.0, 10.0])])
        ground = PrimitiveList([segment(0, [16.0, 10.5], [4.0, 10.5])])
        assert compute(aerial, ground, min_in_patch=1).inlier_pairs == [(0, 2)]

    def test_distance_gate_rejects(self):
        aerial = box_submap() + PrimitiveList([segment(2, [2.0, 10.0], [18.0, 10.0])])
        ground = PrimitiveList([segment(0, [4.0, 13.0], [16.0, 13.0])])
        result = compute(aerial, ground, min_in_patch=1)
        assert (result.n_in_patch, result.n_inliers) == (1, 0)

    def test_disjoint_collinear_lines_are_outliers(self):
        """The classic degenerate slide along a road: collinear, so the distance
        and angle gates alone would accept it."""
        aerial = box_submap(extent=40.0) + PrimitiveList(
            [segment(2, [0.0, 10.0], [10.0, 10.0])]
        )
        ground = PrimitiveList([segment(0, [25.0, 10.0], [35.0, 10.0])])
        result = compute(aerial, ground, min_in_patch=1, line_min_overlap=0.0)
        assert (result.n_in_patch, result.n_inliers) == (1, 0)

    def test_extent_overlap_gate_rejects_barely_overlapping_lines(self):
        aerial = box_submap(extent=40.0) + PrimitiveList(
            [segment(2, [0.0, 10.0], [10.0, 10.0])]
        )
        ground = PrimitiveList([segment(0, [9.0, 10.0], [19.0, 10.0])])  # IoM = 0.1
        assert compute(aerial, ground, min_in_patch=1).n_inliers == 0
        assert compute(aerial, ground, min_in_patch=1, line_min_overlap=0.05).n_inliers == 1

    def test_overlap_is_intersection_over_minimum(self):
        """A short ground fragment lying fully alongside a long aerial road
        scores full overlap (IoM, not IoU)."""
        aerial = box_submap() + PrimitiveList([segment(2, [0.0, 10.0], [20.0, 10.0])])
        ground = PrimitiveList([segment(0, [9.0, 10.0], [11.0, 10.0])])
        result = compute(aerial, ground, min_in_patch=1)
        assert result.n_inliers == 1
        assert result.mean_inlier_quality == pytest.approx(1.0)

    def test_unbounded_lines_are_judged_inside_the_patch(self):
        """Two infinite lines 20 m apart across the patch meet ~1.1 km away at
        1 degree. Unbounded geometry is clipped to the patch, so that far-off
        intersection must not register as an alignment."""
        aerial = box_submap(extent=40.0) + PrimitiveList(
            [line(2, [0.0, 10.0], [1.0, 0.0])]
        )
        d = np.array([np.cos(np.deg2rad(1.0)), np.sin(np.deg2rad(1.0))])
        far = PrimitiveList([line(0, [0.0, 30.0], d)])
        near = PrimitiveList([line(0, [0.0, 10.5], d)])
        assert compute(aerial, far, min_in_patch=1).n_inliers == 0
        assert compute(aerial, near, min_in_patch=1).n_inliers == 1

    def test_ray_pointing_away_does_not_overlap(self):
        """A ray extends from its endpoint along +direction only, so a ray
        pointing away from the aerial road must not score as overlapping it."""
        aerial = box_submap(extent=40.0) + PrimitiveList(
            [segment(2, [0.0, 10.0], [10.0, 10.0])]
        )
        toward = PrimitiveList([ray(0, [30.0, 10.0], [-1.0, 0.0])])
        away = PrimitiveList([ray(0, [30.0, 10.0], [1.0, 0.0])])
        assert compute(aerial, toward, min_in_patch=1).n_inliers == 1
        assert compute(aerial, away, min_in_patch=1).n_inliers == 0


class TestFitnessValue:
    def test_small_support_scores_below_large_support_at_equal_ratio(self):
        """Wilson lower bound: a perfect 1/1 must not outrank a perfect 40/40."""
        patch = box_submap(extent=40.0)
        row = lambda n: PrimitiveList(  # noqa: E731
            [PointPrimitive(id=i, point=np.array([float(i), 0.0])) for i in range(n)]
        )
        f_few = compute(patch + row(1), row(1), min_in_patch=1)
        f_many = compute(patch + row(40), row(40), min_in_patch=1)
        assert f_few.inlier_ratio == f_many.inlier_ratio == 1.0
        assert f_few.fitness < f_many.fitness < 1.0

    def test_below_min_in_patch_scores_zero(self):
        aerial = box_submap()
        aerial.append(PointPrimitive(id=2, point=np.array([10.0, 10.0])))
        ground = PrimitiveList([PointPrimitive(id=0, point=np.array([10.0, 10.0]))])
        result = compute(aerial, ground, min_in_patch=10)
        assert result.n_inliers == 1 and result.fitness == 0.0

    def test_empty_aerial_map_scores_zero(self):
        ground = PrimitiveList([PointPrimitive(id=0, point=np.array([0.0, 0.0]))])
        result = compute(PrimitiveList(), ground)
        assert (result.fitness, result.n_in_patch) == (0.0, 0)

    def test_transform_is_applied_to_the_ground_map(self, real_size_submaps):
        aerial, ground, T = real_size_submaps
        aligned = compute(aerial, ground, T)
        misaligned = compute(aerial, ground, np.eye(4))
        assert aligned.fitness > misaligned.fitness

    def test_rejects_3d_primitives(self):
        aerial = PrimitiveList([PointPrimitive(id=0, point=np.zeros(3))])
        with pytest.raises(ValueError, match="2D matching frame"):
            compute(aerial, aerial)


def test_runtime_at_real_submap_size(real_size_submaps):
    """The matcher scores every clustered hypothesis, so this runs tens of times
    per ground submap. The pairwise geometry is vectorized to keep it in the
    single-digit-millisecond range; the pre-vectorization implementation took
    ~100 ms here.
    """
    aerial, ground, T = real_size_submaps
    assert len(aerial) > 200 and len(ground) > 150
    compute(aerial, ground, T)  # warm up
    elapsed_ms = min(
        (lambda t0: (compute(aerial, ground, T), time.perf_counter() - t0)[1])(
            time.perf_counter()
        )
        for _ in range(5)
    ) * 1e3
    assert elapsed_ms < 25.0, f"alignment fitness took {elapsed_ms:.1f} ms"
