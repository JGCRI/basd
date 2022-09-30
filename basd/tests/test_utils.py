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

    def test_aggregate_periodic(self):
        # [0, 1, 2, 3, 4]
        simple_array = np.arange(5)
        # Take window to include one entry on either side of center
        halfwin = 1

        # Taking the max of a running window of size 3, that wraps,
        # [4, 2, 3, 4, 4]
        simple_max = util.aggregate_periodic(simple_array, halfwin, 'max')
        np.testing.assert_array_equal(simple_max,
                                      np.array([4, 2, 3, 4, 4]))

        # Taking the mean of a running window of size 3, that wraps,
        # [5/3, 1, 2, 3, 7/3]
        simple_mean = util.aggregate_periodic(simple_array, halfwin)
        np.testing.assert_array_almost_equal(simple_mean,
                                             np.array([5/3, 1, 2, 3, 7/3]))

        # NA values
        # The middle three aggregated values will now be nan
        # [0, 1, nan, 3, 3]
        na_array = np.array([0, 1, np.nan, 3, 3])
        na_mean = util.aggregate_periodic(na_array, halfwin)
        np.testing.assert_array_almost_equal(na_mean,
                                             np.array([4/3, np.nan, np.nan, np.nan, 2]))

        # Invalid aggregation method raises value error
        self.assertRaises(ValueError, util.aggregate_periodic,
                          simple_array, halfwin, 'mode')

    def test_get_upper_bound_climatology(self):
        # Number of years in our data
        n_years = 2
        # Data array
        data_arr = np.random.uniform(0, 1, 366 * n_years)
        # Day of the year
        days = np.repeat(np.arange(366) + 1, n_years)
        # Half window size
        halfwin = 15

        # Get upper bounds
        ubcs = util.get_upper_bound_climatology(data_arr, days, halfwin)

        # Assert that shapes are the same for now, can figure out better tests later
        assert ubcs.shape == data_arr.shape

    def test_ccs_transfer_sim2obs_upper_bound_climatology(self):
        # Number of years in our data
        n_years = 2

        # Data arrays
        data_dict = {
            'obs_hist': np.random.uniform(0, 1, 366 * n_years),
            'sim_hist': np.random.uniform(0, 1, 366 * n_years),
            'sim_fut': np.random.uniform(0, 1, 366 * (1+n_years))
        }

        # Associated days
        days = {
            'obs_hist': np.repeat(np.arange(366) + 1, n_years),
            'sim_hist': np.repeat(np.arange(366) + 1, n_years),
            'sim_fut': np.repeat(np.arange(366) + 1, (1+n_years))
        }

        # Transfer climatology trend
        sim_fut_ba_ubc = util.ccs_transfer_sim2obs_upper_bound_climatology(data_dict, days)

        # Assert that shapes are the same for now, can figure out better tests later
        assert sim_fut_ba_ubc.shape == data_dict['sim_fut'].shape

    def test_scale_by_upper_bound_climatology(self):
        # Data array of positive values
        data_arr = np.array([1, 2, 3, 4, 5])
        # Upper bounds
        ubc = np.array([2, 3, 4, 5, 6])
        # Data array as proportion of upper bounds
        data_arr_scaled = data_arr/ubc

        # Everything here is the simple case, so should just divide
        # and then multiply to scale back up
        simple_divide = data_arr/ubc
        simple_prod = data_arr_scaled * ubc
        function_divide = util.scale_by_upper_bound_climatology(data_arr, ubc)
        function_prod = util.scale_by_upper_bound_climatology(data_arr_scaled, ubc, divide=False)

        # Equal in the simple case
        np.testing.assert_array_equal(simple_prod, function_prod)
        np.testing.assert_array_equal(simple_divide, function_divide)

        # What if we have zeros in ubc or data_arr exceeds ubc
        ubc = np.array([0, 1, 4, 5, 6])

        #  should get [0, 1, 3/4, 4/5, 5/6]
        function_divide = util.scale_by_upper_bound_climatology(data_arr, ubc)
        np.testing.assert_array_almost_equal(np.array([0, 1, 3/4, 4/5, 5/6]),
                                             function_divide)

        # nans
        data_arr = np.array([np.nan, 2, 4])
        ubc = np.array([np.nan, np.nan, 3])
        nan_divide = util.scale_by_upper_bound_climatology(data_arr, ubc)
        np.testing.assert_array_almost_equal(np.array([np.nan, np.nan, 1]),
                                             nan_divide)


if __name__ == '__main__':
    unittest.main()
