import os
import unittest

import pandas as pd
import xarray as xr

from basd import Downscaler
from basd import Parameters
import basd.sd_utils as sdu


class StatisticalDownscalingTest(unittest.TestCase):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(THIS_DIR, '../data')
    PR_OBS_HIST = xr.open_dataset(os.path.join(DATA_PATH,
                                               'pr_obs-hist_fine_1979-2014.nc'))
    PR_SIM_FUT = xr.open_dataset(os.path.join(DATA_PATH,
                                              'pr_sim-fut_coarse_2065-2100.nc'))
    PR_PARAMS = Parameters(lower_bound=0, lower_threshold=0.0000011574,
                           trend_preservation='mixed', distribution='gamma',
                           if_all_invalid_use=0)


if __name__ == '__main__':
    unittest.main()
