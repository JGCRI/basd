import os
import timeit
import unittest

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

import basd
import basd.utils as util


class TestStatisticalDownscaling(unittest.TestCase):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(THIS_DIR, '../data')

    def test_grid_cell_weights(self):
        lats = np.arange(-90, 90)

    def test_get_data_at_loc(self):
        # Function input
        loc = tuple((0, 0))
        fine_arr = np.array([
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
            [[19, 20], [21, 22], [23, 24]]
        ])
        obs_fine = fine_arr.copy().reshape(3, 2, 4)
        print(obs_fine)
        sim_fine = fine_arr.copy().transpose(1, 2, 0)
        sim_coarse = np.array([
            [[1]],
            [[2]],
            [[3]],
            [[4]]
        ])
        month_numbers = {
            'obs_fine': np.arange(0, 4),
            'sim_coarse': np.arange(0, 4)
        }
        downscaling_factors = {
            'lat': 3,
            'lon': 2
        }
        sum_weights = np.array([
            [[1, 1], [2, 2], [3, 3]]
        ])

        # Apply function
        data_loc, sum_weights_loc = basd.get_data_at_loc(loc,
                                                         obs_fine, sim_coarse, sim_fine,
                                                         month_numbers, downscaling_factors,
                                                         sum_weights)

        # What the output should look like
        test_output_arr = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24]
        ])
        test_output_weights = np.array([1, 1, 2, 2, 3, 3])

        # Test assertion
        assert np.array_equal(data_loc['obs_fine'], test_output_arr)
        assert np.array_equal(sum_weights_loc, test_output_weights)
        assert np.array_equal(obs_fine, data_loc['obs_fine'].T.reshape(2, 2, 4))

    def test_downscale_chunk(self):
        rot_mats = basd.generate_cre_matrix(4)
        fine_arr = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]]
        ])
        obs_fine = fine_arr.copy().transpose(1, 2, 0)
        sim_fine = fine_arr.copy().transpose(1, 2, 0)
        sim_coarse = np.array([
            [[1]],
            [[2]],
            [[3]],
            [[4]]
        ])
        month_numbers = {
            'obs_fine': np.arange(0, 4),
            'sim_coarse': np.arange(0, 4)
        }
        downscaling_factors = {
            'lat': 2,
            'lon': 2
        }
        sum_weights = np.array([
            [[1, 1], [2, 2]]
        ])
        params = basd.Parameters()
        basd.downscale_chunk(obs_fine, sim_coarse, sim_fine,
                             sum_weights, params, month_numbers,
                             downscaling_factors, rot_mats)

    def test_downscale_one_location_parallel(self):
        obs_fine_path = 'rsds_obs-hist_fine_1979-2014.nc'
        sim_coarse_path = 'rsds_sim-fut_coarse_2065-2100.nc'

        obs_fine = xr.open_dataset(os.path.join(self.DATA_PATH, obs_fine_path))
        sim_coarse = xr.open_dataset(os.path.join(self.DATA_PATH, sim_coarse_path))

        tot_size = obs_fine.sizes['lat'] * obs_fine.sizes['lon'] * obs_fine.sizes['time']
        new_data = np.arange(tot_size).reshape((obs_fine.sizes['time'],
                                                obs_fine.sizes['lat'],
                                                obs_fine.sizes['lon'])).transpose((1, 2, 0))
        obs_fine['rsds'].data = new_data

        obs_fine = obs_fine.transpose('lat', 'lon', 'time')
        sim_coarse = sim_coarse.transpose('lat', 'lon', 'time')

        params = basd.Parameters(halfwin_ubc=15,
                                 trend_preservation='bounded',
                                 distribution='beta',
                                 lower_bound=0,
                                 upper_bound=1,
                                 lower_threshold=0.0001,
                                 upper_threshold=0.9999,
                                 if_all_invalid_use=0)

        sd = basd.Downscaler(obs_fine, sim_coarse, 'rsds', params)

        days, months, years = basd.time_scraping(sd.datasets)
        downscaling_factors = {
            'lat': 2,
            'lon': 2
        }
        rotation_matrices = []
        weights = sd.sum_weights.repeat(4).reshape(4, 4, 1)

        sd.obs_fine['rsds'] = sd.obs_fine['rsds'].transpose('lat', 'lon', 'time')
        sd.sim_fine['rsds'] = sd.sim_fine['rsds'].transpose('lat', 'lon', 'time')
        sd.sim_coarse['rsds'] = sd.sim_coarse['rsds'].transpose('lat', 'lon', 'time')

        print(sd.sim_fine['rsds'].data.shape)

        data, long_weights = basd.get_data_at_loc((0, 0),
                                                  sd.obs_fine['rsds'].data,
                                                  sd.sim_coarse['rsds'].data,
                                                  sd.sim_fine['rsds'].data,
                                                  months, downscaling_factors, weights)

        print(data['obs_fine'])

        # out_chunk = basd.downscale_chunk(sd.obs_fine['rsds'].data,
        #                                  sd.sim_coarse['rsds'].data,
        #                                  sd.sim_fine['rsds'].data,
        #                                  weights, params, months, downscaling_factors, rotation_matrices)
        #
        # print(out_chunk)
        # print(sd.obs_fine['rsds'][dict(time=0, lat=0)].values)
