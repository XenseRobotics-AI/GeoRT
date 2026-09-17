import unittest
import numpy as np
from geort.evaluate_differences import lp_filter, disjoint_pairs, response


class DifferenceAuditTests(unittest.TestCase):
    def test_filter_first_frame_and_clip_reset(self):
        q = np.array([[2.], [12.], [12.]])
        np.testing.assert_allclose(lp_filter(q).ravel(), [2., 5., 7.1])
        np.testing.assert_array_equal(lp_filter(q[1:]), q[1:])
        for bad in (np.empty((0, 1)), np.array([[np.nan]])):
            with self.assertRaises(ValueError): lp_filter(bad)

    def test_pair_selection_has_no_reused_endpoints_and_is_order_invariant(self):
        p = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [4, 5], [5, 5]])
        selected = disjoint_pairs(p)
        self.assertEqual(len(np.unique(selected)), selected.size)
        np.testing.assert_array_equal(selected, disjoint_pairs(p[::-1]))
        self.assertEqual(disjoint_pairs([]).shape, (0, 2))

    def test_no_motion_is_counted_as_low_response_not_good_direction(self):
        h = np.array([[.001, 0, 0], [.001, 0, 0]])
        r = np.array([[0., 0, 0], [-.001, 0, 0]])
        result = response(h, r)
        self.assertEqual(result['low_response_fraction'], .5)
        self.assertEqual(result['direction_deg_responsive_only']['n'], 1)
        self.assertEqual(result['direction_deg_responsive_only']['mean'], 180.)
        with self.assertRaises(ValueError): response(np.zeros((1, 3)), r[:1])
