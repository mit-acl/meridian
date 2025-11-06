import clipperpy

from roman.align.object_registration import ObjectRegistration

from gen_seg_match.segment.line_cloud import LineCloud

LineSegmentMatcherParams = clipperpy.invariants.LineSegmentDistParams


class LineSegmentMatcher(ObjectRegistration):
    def __init__(self, params: LineSegmentMatcherParams):
        super().__init__(dim=params.dim)
        self.params = params
        return

    def _setup_clipper(self):
        invariant = clipperpy.invariants.LineSegmentDist(self.params)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)
        return clipper

    def _clipper_score_all_to_all(self, clipper, lcd1: LineCloud, lcd2: LineCloud):
        A_init = clipperpy.utils.create_all_to_all(len(lcd1), len(lcd2))

        # self._check_clipper_arrays()

        clipper.score_pairwise_consistency(
            lcd1.centers_and_directions().T, lcd2.centers_and_directions().T, A_init
        )
        return clipper, A_init

    # def _check_clipper_arrays(self, map1_cl, map2_cl):
    #     assert map1_cl.shape[1] == map2_cl.shape[1]
    #     # TODO: check that the number of point elements + feature elements is correct
    #     # if self.use_gravity:
    #     #     assert map1_cl.shape[1] == 3 + 2, f"map1_cl.shape[1] = {map1_cl.shape[1]}"
    #     return
