import numpy as np
import pandas as pd

import basd.ba_params as bap
import basd.utils as util


class Adjustment:
    def __init__(self, obs_hist, sim_hist, sim_fut, variable, params):

        # Setting the data
        self.obs_hist = obs_hist
        self.sim_hist = sim_hist
        self.sim_fut = sim_fut
        self.sim_fut_ba = None
        self.variable = variable
        self.params = params

        # TODO: Assert that input data has same spatial dimension
        # coords = util.analyze_input_nc(obs_hist, variable)

        # TODO: Assert full period coverage if using running window mode
        # TODO: Assert uniform number of days between input data
        # TODO: Abort if there are only missing values in any of the data input
        # TODO: Scale data if halfwin_upper_bound_climatology

    def assert_consistency_of_data_resolution(self):
        """
        Raises an assertion error if any of the input data are not of the same resolution
        """
        # TODO: maybe relax this assertion so that they are a multiple, and aggregate to correct resolution

        coords = {
            'obs_hist': util.analyze_input_nc(self.obs_hist, self.variable),
            'sim_hist': util.analyze_input_nc(self.sim_hist, self.variable),
            'sim_fut': util.analyze_input_nc(self.sim_fut, self.variable)
        }

        # TODO: This only works if in the NetCDF file, latitude and longitude are written as 'lat', 'lon'
        assert coords.get('obs_hist').get('lat').size == coords.get('sim_hist').get('lat').size == coords.get(
            'sim_fut').get('lat').size
        assert coords.get('obs_hist').get('lon').size == coords.get('sim_hist').get('lon').size == coords.get(
            'sim_fut').get('lon').size

    def adjust_bias_one_location(self, i_loc):
        """
        Bias adjusts one grid cell

        Parameters
        ----------
        self: Adjustment
            Bias adjustment object
        i_loc: tuple
            index of grid cell to bias adjust

        Returns
        -------
        sim_fut_ba_loc: xarray.DataArray
            adjusted time series with times, lat and lon
        """
        # Get data at one location
        obs_hist_loc = self.obs_hist[self.variable][i_loc]
        sim_hist_loc = self.sim_hist[self.variable][i_loc]
        sim_fut_loc = self.sim_fut[self.variable][i_loc]

        # Put in dictionary for easy iteration
        data_loc = {
            'obs_hist': obs_hist_loc,
            'sim_hist': sim_hist_loc,
            'sim_fut': sim_fut_loc
        }

        # Get long term mean over each time series using valid values
        long_term_mean = {
            'obs_hist': util.average_valid_values(obs_hist_loc.values, np.nan,
                                                  self.params.lower_bound, self.params.lower_threshold,
                                                  self.params.upper_bound, self.params.upper_threshold),
            'sim_hist': util.average_valid_values(sim_hist_loc.values, np.nan,
                                                  self.params.lower_bound, self.params.lower_threshold,
                                                  self.params.upper_bound, self.params.upper_threshold),
            'sim_fut': util.average_valid_values(sim_fut_loc.values, np.nan,
                                                 self.params.lower_bound, self.params.lower_threshold,
                                                 self.params.upper_bound, self.params.upper_threshold)
        }

        # Scraping the time from the data and turning into pandas date time array
        days, month_numbers, years = util.time_scraping(self)

        # TODO: Implement option for month-to-month bias adjustment
        # Get window centers for running window mode
        if self.params.step_size:
            window_centers = util.window_centers_for_running_bias_adjustment(days, self.params.step_size)
        else:
            msg = 'Month to month bias adjustment not yet implemented. Set step_size to be non-zero.'
            raise Exception(msg)

        # Result
        result = sim_fut_loc.copy

        # TODO: Implement month to month adjustment
        # Adjust bias for each window center
        for window_center in window_centers:
            data_this_window, years_this_window = util.get_data_in_window(window_center, data_loc,
                                                                          days, years,
                                                                          long_term_mean)

            # Send data to adjust bias one month
            result_this_window = util.adjust_bias_one_month(data_this_window, years_this_window,
                                                            long_term_mean, self.params)

            # put central part of bias-adjusted data into result
            m_ba = util.window_indices_for_running_bias_adjustment(days['sim_fut'], window_center, 31)
            m_keep = util.window_indices_for_running_bias_adjustment(days['sim_fut'], window_center,
                                                                     self.params.step_size, years['sim_fut'])
            m_ba_keep = np.in1d(m_ba, m_keep)
            # TODO: Why are we saving some of result and some of the input?
            result[m_keep] = result_this_window[m_ba_keep]

            # TODO: How should we return. This is going to be a time series for every day
            #   in sim_fut. Thus we can create a variable equivalent to sim_fut_loc, but put
            #   these results into the values
            sim_fut_ba_loc = sim_fut_loc.copy()

            sim_fut_ba_loc.values = result

        return sim_fut_ba_loc
