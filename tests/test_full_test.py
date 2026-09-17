import unittest
import numpy as np
from geort.evaluate_full_test import response_summary


class FullTestAggregationTests(unittest.TestCase):
    def test_all_reads_count_and_stationary_not_scored_as_direction(self):
        h=np.array([[[.001,0,0],[0,0,0],[.02,0,0]]])
        result=response_summary(h,h)
        self.assertEqual(result['all_transition_finger_count'],3)
        self.assertEqual(result['moving_ge_point1mm']['n'],2)
        self.assertEqual(result['fine_point5_to10mm']['n'],1)
        self.assertEqual(result['stationary_proxy_mm']['n'],1)
        self.assertEqual(result['moving_ge_point1mm']['direction_deg_responsive_only']['mean'],0)

    def test_unresponsive_outputs_remain_in_denominator(self):
        h=np.array([[[.001,0,0],[.002,0,0]]]);r=np.zeros_like(h)
        result=response_summary(h,r)['moving_ge_point1mm']
        self.assertEqual(result['n'],2);self.assertEqual(result['low_response_fraction'],1)
        self.assertIsNone(result['direction_deg_responsive_only'])
        self.assertEqual(result['gain_deviation_from_one']['mean'],1)

    def test_empty_amplitude_band_is_missing_not_zero_error(self):
        x=np.zeros((2,5,3));result=response_summary(x,x)
        self.assertIsNone(result['moving_ge_point1mm'])
        self.assertIsNone(result['fine_point5_to10mm'])
        self.assertEqual(result['stationary_proxy_mm']['n'],10)


if __name__=='__main__':unittest.main()
