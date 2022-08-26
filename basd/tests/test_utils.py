import unittest

import numpy as np
import numpy.ma as ma
import pandas as pd

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

        # TODO: Don't really understand this.
        #       Seems like days should be between 1-365 (or maybe 366),
        #       but we have them in 1-31. Maybe this should change depending on modes
        # Should return indexes where we want data for running window mode
        indexes = util.window_indices_for_running_bias_adjustment(days, window_center, window_width)
        # Starting at the second day of the year:
        # TODO: This only makes sense if we have data for full years
        #   indexes for 2019: -14 -- 16 which in turn will be 0 -- 16, and 442 -- 455
        #   indexes for 2020: 351=366-15 -- 366+155=381
        #   This is why we assert full time coverage earlier in the run?
        #   Fill empty values if the full period isn't covered?
        print(indexes)
        print(days.size)


if __name__ == '__main__':
    unittest.main()
