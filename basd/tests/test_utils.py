import random
import unittest

import numpy as np
import numpy.ma as ma
import pandas as pd

import basd
import basd.utils as util


class TestUtils(unittest.TestCase):

    COMP_ARRAY = np.array([1, 2, 3])
    COMP_MASKED_ARRAY_FLOAT = ma.masked_array(np.array([1, 2., 3]), mask=[0, 1, 0])

    def test_ma2a(self):
        """
        Makes sure masked arrays are filled with NaNs when masked. Only for float arrays.
        Raise error flag must be set to True for integer arrays. When no mask, should return
        the input array.
        """
        nonthinking = util.ma2a(self.COMP_ARRAY, raise_error=True)
        maskmiddlefloat = util.ma2a(self.COMP_MASKED_ARRAY_FLOAT)

        np.testing.assert_array_equal(self.COMP_ARRAY, nonthinking)
        np.testing.assert_array_equal(np.array([1, np.nan, 3]), maskmiddlefloat)

    def test_average_valid_values(self):
        """
        Confirms that the correct averages are produced. Input array and average over
        valid values. Values are set to respective bounds when beyond input thresholds,
        which are optional.
        """
        # Simple average of [1, 2, 3], which is 2
        avgeasy = util.average_valid_values(self.COMP_ARRAY)
        assert avgeasy == 2

        # Masks out second entry. Should be average of [1, -, 3], which is 2
        avgmask = util.average_valid_values(self.COMP_MASKED_ARRAY_FLOAT)
        assert avgmask == 2

        # Should set array to [0, 2, 10] with avg value of 4
        avgbound = util.average_valid_values(self.COMP_ARRAY, lower_bound=0, lower_threshold=1.5,
                                             upper_bound=10, upper_threshold=2.5)
        assert avgbound == 4

    def test_window_indices_for_running_bias_adjustment(self):
        # Second day of the year
        window_center = 2
        # Window width 31 days
        window_width = 31
        # Artificial date range, has to start at Jan 1st and end Dec 31st of some years
        dates = pd.date_range(start='1/1/2019', end='12/31/2020', freq='D')
        # Day of the month for each date
        days = dates.day_of_year
        # Year of each date
        years = dates.year

        # Should return indexes where we want data for running window mode
        indexes = util.window_indices_for_running_bias_adjustment(days, window_center, window_width)

        # TODO: This only makes sense if we have data for full years
        #   This is why we assert full time coverage earlier in the run?
        correct = np.concatenate((np.arange(0, 17), np.arange(351, 382), np.arange(717, 731)))

        # Assertion of equality
        np.testing.assert_array_equal(indexes, correct)

    def test_ccs_transfer_sim2obs(self):
        small = 0.25
        med = 0.5
        big = 0.75

        zerobias = util.ccs_transfer_sim2obs(med, med, big)
        negbias = util.ccs_transfer_sim2obs(med, small, big)
        negbiasadd = util.ccs_transfer_sim2obs(big, med, small)
        posbias = util.ccs_transfer_sim2obs(small, big, med)
        posbiasadd = util.ccs_transfer_sim2obs(small, med, big)

        assert zerobias == 0.75
        assert negbias == 1-0.5/3
        assert negbiasadd == 0.5
        assert posbias == 0.5/3
        assert posbiasadd == 0.5

    def test_extreme_value_probabilities(self):
        seed = np.random.RandomState(1)
        data = {
            'obs_hist': seed.weibull(1, 10),
            'sim_hist': seed.weibull(1.5, 10),
            'sim_fut': seed.weibull(2, 10)
        }

        # Lower bound only
        params = basd.Parameters(lower_bound=0, lower_threshold=0.1,
                                 unconditional_ccs_transfer=True)
        lower = True
        upper = False
        plt, put, plout = util.extreme_value_probabilities(data, params, lower, upper)
        assert plt == 0.1
        assert put is None
        assert plout is None

        # Lower and upper bound and no CCS transfer
        params = basd.Parameters(lower_bound=0, lower_threshold=0.1,
                                 upper_bound=1000, upper_threshold=999,
                                 unconditional_ccs_transfer=False)
        upper = True
        plt, put, plout = util.extreme_value_probabilities(data, params, lower, upper)
        assert plt == 0.1
        assert put == 0
        assert plout == 0.1

        # Upper bound only
        params = basd.Parameters(upper_bound=1000, upper_threshold=999,
                                 unconditional_ccs_transfer=True)
        lower = False
        plt, put, plout = util.extreme_value_probabilities(data, params, lower, upper)
        assert plt is None
        assert put == 0
        assert plout is None

        # No bounds
        params = basd.Parameters()
        upper = False
        plt, put, plout = util.extreme_value_probabilities(data, params, lower, upper)
        assert plt is None
        assert put is None
        assert plout is None


if __name__ == '__main__':
    unittest.main()
