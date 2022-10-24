import os
import unittest

import xarray as xr

import basd


class Profiling(unittest.TestCase):

    DATA_PATH = '/Users/prim232/OneDrive - PNNL/Documents/BASD/data'

    def test_rsds_profiling(self):
        rsds_obs_hist_path = 'rsds_ewembi_2011_2016.nc4'
        rsds_sim_hist_path = 'rsds_day_MIROC6_historical_r2i1p1f1_gn_20100101-20141231.nc'
        rsds_sim_fut_path = 'rsds_day_MIROC6_ssp370_r2i1p1f1_gn_20150101-20241231.nc'

        rsds_obs_hist = xr.open_dataset(os.path.join(self.DATA_PATH, rsds_obs_hist_path),
                                        decode_coords='all')
        rsds_sim_hist = xr.open_dataset(os.path.join(self.DATA_PATH, rsds_sim_hist_path),
                                        decode_coords='all')
        rsds_sim_fut = xr.open_dataset(os.path.join(self.DATA_PATH, rsds_sim_fut_path),
                                       decode_coords='all')

        rsds_params = basd.Parameters(halfwin_ubc=15,
                                      trend_preservation='bounded',
                                      distribution='beta',
                                      lower_bound=0,
                                      upper_bound=1,
                                      lower_threshold=0.0001,
                                      upper_threshold=0.9999,
                                      if_all_invalid_use=0)

        rsds_ba = basd.Adjustment(rsds_obs_hist,
                                  rsds_sim_hist,
                                  rsds_sim_fut,
                                  'rsds',
                                  rsds_params,
                                  remap_grid=True)

        loc = dict(lat=100, lon=100)
        rsds_ba.adjust_bias_one_location(loc)


if __name__ == '__main__':
    unittest.main()
