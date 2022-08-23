import unittest

import numpy as np
import numpy.ma as ma

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


if __name__ == '__main__':
    unittest.main()
