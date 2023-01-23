import datetime as dt
import os

import dask.array as da
import numpy as np
import xarray as xr

from basd.ba_params import Parameters
import basd.regridding as rg
import basd.utils as util
import basd.one_loc_output as olo


class Adjustment:
    def __init__(self,
                 obs_hist: xr.Dataset,
                 sim_hist: xr.Dataset,
                 sim_fut: xr.Dataset,
                 variable: str,
                 params: Parameters,
                 remap_grid: bool = False):

        # Setting the data
        # Also converting calendar to proleptic_gregorian.
        # See xarray.convert_calendar() fro details, but default uses date alignment
        # for 360_day calendars, as this preserves month data for month-to-month mode
        self.obs_hist = obs_hist.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
        self.sim_hist = sim_hist.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
        self.sim_fut = sim_fut.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
        self.variable = variable
        self.params = params
        self.input_calendar = sim_hist.time.dt.calendar
        self.datasets = {
            'obs_hist': self.obs_hist,
            'sim_hist': self.sim_hist,
            'sim_fut': self.sim_fut
        }

        # Set dimension names to lat, lon, time
        self.datasets = util.set_dim_names(self.datasets)
        self.obs_hist = self.datasets['obs_hist']
        self.sim_hist = self.datasets['sim_hist']
        self.sim_fut = self.datasets['sim_fut']

        # Maps observational data onto simulated data grid resolution
        if remap_grid:
            # self.obs_hist = rg.match_grids(self.obs_hist, self.sim_hist, self.sim_fut)
            self.obs_hist = rg.project_onto(self.obs_hist, self.sim_hist, self.variable)
            self.datasets['obs_hist'] = self.obs_hist

        # Forces data to have same spatial shape and resolution
        self.assert_consistency_of_data_resolution()
        self.sizes = self.obs_hist.sizes

        # Assert full period coverage if using running window mode
        if params.step_size:
            self.assert_full_period_coverage()

        # TODO: Assert uniform number of days between input data
        # TODO: Abort if there are only missing values in any of the data input
        # TODO: Scale data if halfwin_upper_bound_climatology

        # Set up output dataset to have same form as input now that it has been standardized
        self.sim_fut_ba = self.sim_fut.copy()

    def assert_full_period_coverage(self):
        """
        Raises an assertion error if years aren't fully covered. Trims data to the first Jan 1st
        available to the last Dec 31st available.
        """
        for key, data in self.datasets.items():
            # Trim data to start Jan 1 and end Dec 31
            # Indexes of each Jan 1st and Dec 31st
            jan_first = data.time.dt.strftime("%m-%d") == '01-01'
            dec_thirty_first = data.time.dt.strftime("%m-%d") == '12-31'
            # Index of first Jan 1st and last Dec 31st
            first = min([i for i, x in enumerate(jan_first) if x])
            last = max([i for i, x in enumerate(dec_thirty_first) if x])
            # Indexes to keep
            keep = [((i <= last) and (i >= first)) for i in range(jan_first.size)]
            # Selecting data
            self.datasets[key] = data.sel(time=keep)

        # Updating data
        self.obs_hist = self.datasets['obs_hist']
        self.sim_hist = self.datasets['sim_hist']
        self.sim_fut = self.datasets['sim_fut']

        # Getting updated time info
        days, month_numbers, years = util.time_scraping(self.datasets)

        # Asserting all years are here within range of first and last
        for key, data in self.datasets.items():
            # Make sure years array is continuous
            year_arr = years[key]
            years_sorted_unique = np.unique(year_arr)
            ys = years_sorted_unique[0]
            ye = years_sorted_unique[-1]
            msg = f'Not all years between {ys} and {ye} are covered in {key}'
            assert years_sorted_unique.size == ye - ys + 1, msg

            # Getting every day and year that actually happened within our bounds
            # and making sure those are all present in our arrays
            day_arr = days[key]
            years_true = []
            days_true = []
            for year in years_sorted_unique:
                n_days = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days
                years_true.append(np.repeat(year, n_days))
                days_true.append(np.arange(1, n_days + 1))
            years_true = np.concatenate(years_true)
            days_true = np.concatenate(days_true)

            year_arr = year_arr.to_numpy()
            day_arr = day_arr.to_numpy()

            # make sure all days from ys-01-01 to ye-12-31 are covered
            msg = f'not all days between {ys}-01-01 and {ye}-12-31 are covered in {key}'
            assert year_arr.size == years_true.size and day_arr.size == days_true.size, msg
            assert np.all(year_arr == years_true) and np.all(day_arr == days_true), msg

    def assert_consistency_of_data_resolution(self):
        """
        Raises an assertion error if data are not of the same shape or cannot be aggregated to
        the same resolution. Otherwise, it will force data to have same spatial resolution via
        aggregation.
        """
        # Get the most coarse dimensions
        min_lat = min([self.obs_hist.sizes.get('lat'), self.sim_hist.sizes.get('lat'), self.sim_fut.sizes.get('lat')])
        min_lon = min([self.obs_hist.sizes.get('lon'), self.sim_hist.sizes.get('lon'), self.sim_fut.sizes.get('lon')])

        # For each dataset, aggregate to the most coarse dataset if possible
        for key, value in self.datasets.items():
            agg_lat = value.sizes.get('lat') / min_lat
            agg_lon = value.sizes.get('lon') / min_lon
            assert agg_lat == agg_lon, f'Data have differing shapes'
            assert agg_lat.is_integer(), f'Non-integer aggregation factor for {key}'
            agg_fact = int(agg_lat)
            if agg_fact > 1:
                print(f'Aggregating {key} by a factor of {agg_fact}')
                self.datasets[key] = value.coarsen(lat=agg_fact).mean().coarsen(lon=agg_fact).mean()

        # Save aggregated datasets
        self.obs_hist = self.datasets['obs_hist']
        self.sim_hist = self.datasets['sim_hist']
        self.sim_fut = self.datasets['sim_fut']

    def adjust_bias_one_location(self, i_loc, full_details=True):
        """
        Bias adjusts one grid cell

        Parameters
        ----------
        self: Adjustment
            Bias adjustment object
        i_loc: dict
            index of grid cell to bias adjust
        full_details: bool
            Should function return full details of run, or just the time series array

        Returns
        -------
        sim_fut_ba_loc: xarray.DataArray
            adjusted time series with times, lat and lon
        """
        # Get data at one location
        obs_hist_loc = self.obs_hist[self.variable][i_loc]
        sim_hist_loc = self.sim_hist[self.variable][i_loc]
        sim_fut_loc = self.sim_fut[self.variable][i_loc]

        # Scraping the time from the data and turning into pandas date time array
        days, month_numbers, years = util.time_scraping(self.datasets)

        # Put in dictionary for easy iteration
        data_loc = {
            'obs_hist': obs_hist_loc.values,
            'sim_hist': sim_hist_loc.values,
            'sim_fut': sim_fut_loc.values
        }

        # If scaling using climatology, get upper bound for scaling
        ubc_ba = None
        ubcs = None
        if self.params.halfwin_ubc:
            ubcs = {
                'obs_hist': util.get_upper_bound_climatology(obs_hist_loc.values,
                                                             days['obs_hist'],
                                                             self.params.halfwin_ubc),
                'sim_hist': util.get_upper_bound_climatology(sim_hist_loc.values,
                                                             days['sim_hist'],
                                                             self.params.halfwin_ubc),
                'sim_fut': util.get_upper_bound_climatology(sim_fut_loc.values,
                                                            days['sim_fut'],
                                                            self.params.halfwin_ubc)
            }
            for key, value in data_loc.items():
                data_loc[key].values = util.scale_by_upper_bound_climatology(value.values, ubcs[key], divide=True)

            ubc_ba = util.ccs_transfer_sim2obs_upper_bound_climatology(ubcs, days)

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

        # Result, to be updated as adjustment is made
        result = sim_fut_loc.copy()

        # Get window centers for running window mode
        if self.params.step_size:
            window_centers = util.window_centers_for_running_bias_adjustment(days, self.params.step_size)
            result = running_window_mode(result, window_centers, data_loc, days, years, long_term_mean, self.params)
        else:
            result = month_to_month_mode(result, data_loc, month_numbers, years, long_term_mean, self.params)

        # If we scaled variable before, time to scale back
        if self.params.halfwin_ubc:
            result.values = util.scale_by_upper_bound_climatology(result.values,
                                                                  ubc_ba,
                                                                  divide=False)
            obs_hist_loc.values = util.scale_by_upper_bound_climatology(obs_hist_loc.values,
                                                                        ubcs['obs_hist'],
                                                                        divide=False)
            sim_hist_loc.values = util.scale_by_upper_bound_climatology(sim_hist_loc.values,
                                                                        ubcs['sim_hist'],
                                                                        divide=False)
            sim_fut_loc.values = util.scale_by_upper_bound_climatology(sim_fut_loc.values,
                                                                       ubcs['sim_fut'],
                                                                       divide=False)

        # Return resulting array with extra details if requested
        if full_details:
            return olo.BaLocOutput(result, sim_fut_loc, obs_hist_loc, sim_hist_loc, self.variable, self.params)

        # Return just resulting array if extra details not requested
        return result

    def adjust_bias(self, lat_chunk_size: int = 0, lon_chunk_size: int = 0,
                    file: str = None, encoding=None):
        """
        Does bias adjustment at every location of input data

        Parameters
        ----------
        file: str
            Location and name string to save output file
        lat_chunk_size: int
            Number of cells to include in chunk in lat direction
        lon_chunk_size: int
            Number of cells to include in chunk in lon direction
        encoding: dict
            Parameter for to_netcdf function

        Returns
        -------
        sim_fut_ba: DataSet
            Temporal grid of adjusted observations

        """
        # Get days, months and years data
        days, month_numbers, years = util.time_scraping(self.datasets)

        if lat_chunk_size & lon_chunk_size:
            # Manual chunk method
            self.obs_hist = self.obs_hist.chunk(dict(lon=lon_chunk_size, lat=lat_chunk_size, time=-1))
            self.sim_hist = self.sim_hist.chunk(dict(lon=lon_chunk_size, lat=lat_chunk_size, time=-1))
            self.sim_fut = self.sim_fut.chunk(dict(lon=lon_chunk_size, lat=lat_chunk_size, time=-1))

        else:
            # Auto chunk method (not allowing to be chunked over time)
            self.obs_hist = self.obs_hist.chunk(dict(lon=None, lat=None, time=-1))
            self.sim_hist = self.sim_hist.chunk(dict(lon=None, lat=None, time=-1))
            self.sim_fut = self.sim_fut.chunk(dict(lon=None, lat=None, time=-1))

        # Order dimensions lon, lat, time
        self.obs_hist[self.variable] = self.obs_hist[self.variable].transpose('lon', 'lat', 'time')
        self.sim_hist[self.variable] = self.sim_hist[self.variable].transpose('lon', 'lat', 'time')
        self.sim_fut[self.variable] = self.sim_fut[self.variable].transpose('lon', 'lat', 'time')
        self.sim_fut_ba[self.variable] = self.sim_fut_ba[self.variable].transpose('lon', 'lat', 'time')

        # Set up dask computation
        ba_output_data = da.map_blocks(adjust_bias_chunk,
                                       self.obs_hist[self.variable].data,
                                       self.sim_hist[self.variable].data,
                                       self.sim_fut[self.variable].data,
                                       params=self.params,
                                       days=days, month_numbers=month_numbers, years=years,
                                       dtype=object, chunks=self.sim_fut[self.variable].chunks)

        # Compute bias adjustment in chunks
        # ba_output_data.persist()

        # Save output
        self.sim_fut_ba[self.variable].data = ba_output_data

        # If provided a path to save NetCDF file, save adjusted DataSet,
        # else just return the result
        if file:
            self.save_adjustment_nc(file, encoding)
        else:
            return self.sim_fut_ba

    def save_adjustment_nc(self, file, encoding=None):
        """
        Saves adjusted data to NetCDF file at specific path

        Parameters
        ----------
        file: str
            Location and name string to save output file
        encoding: dict
            Parameter for to_netcdf function
        """
        # Make sure we've computed
        self.sim_fut_ba = self.sim_fut_ba.persist()

        # Try converting calendar back to input calendar
        try:
            self.sim_fut_ba = self.sim_fut_ba.convert_calendar(self.input_calendar, align_on='date')
        except AttributeError:
            AttributeError('Unable to convert calendar')

        self.sim_fut_ba.to_netcdf(file, encoding={self.variable: encoding})


def running_window_mode(result, window_centers, data_loc, days, years, long_term_mean, params):
    """
    Performs bias adjustment in running window mode

    Parameters
    ----------
    result: xr.DataArray
        Copy of the data to be adjusted (in 1D xarray)
    window_centers: np.Array
        List of days to center window around
    data_loc: dict
        the three data arrays that we need in the specified location
    days: dict
        array of days for each input data array
    years: dict
        array of years for each input data array
    long_term_mean: dict
        array of means for each input data array
    params: Parameters
        object that holds parameters for bias adjustment

    Returns
    -------
    result: x.DataArray
        1D array containing time series of adjusted values
    """
    # Adjust bias for each window center
    for window_center in window_centers:
        data_this_window, years_this_window = util.get_data_in_window(window_center, data_loc,
                                                                      days, years, long_term_mean, params)

        # Send data to adjust bias one month
        result_this_window = util.adjust_bias_one_month(data_this_window, years_this_window,
                                                        params)

        # put central part of bias-adjusted data into result
        m_ba = util.window_indices_for_running_bias_adjustment(days['sim_fut'], window_center, 31)
        m_keep = util.window_indices_for_running_bias_adjustment(days['sim_fut'], window_center,
                                                                 params.step_size, years['sim_fut'])
        m_ba_keep = np.in1d(m_ba, m_keep)
        result[m_keep] = result_this_window[m_ba_keep]

    return result


def month_to_month_mode(result, data_loc, month_numbers, years, long_term_mean, params):
    """
    Performs bias adjustment in month-to-month mode

    Parameters
    ----------
    result: xr.DataArray
        Copy of the data to be adjusted (in 1D xarray)
    data_loc: dict
        the three data arrays that we need in the specified location
    month_numbers: dict
        array of months for each input data array
    years: dict
        array of years for each input data array
    long_term_mean: dict
        array of means for each input data array
    params: Parameters
        object that holds parameters for bias adjustment

    Returns
    -------
    result: x.DataArray
        1D array containing time series of adjusted values
    """
    # Adjust bias for each window center
    for month in params.months:
        data_this_month, years_this_month = util.get_data_in_month(month, data_loc,
                                                                   years, month_numbers,
                                                                   long_term_mean, params)

        # Send data to adjust bias one month
        result_this_month = util.adjust_bias_one_month(data_this_month, years_this_month, params)

        # put bias-adjusted data into result
        m = month_numbers['sim_fut'] == month
        result[m] = result_this_month

    return result


def adjust_bias_chunk(obs_hist, sim_hist, sim_fut, params, days, month_numbers, years):
    # Iterate through each location and adjust bias
    i_locations = np.ndindex(obs_hist.shape[0], obs_hist.shape[1])

    # Unadjusted results of correct shape
    sim_fut_ba = sim_fut

    # Find and save results into adjusted DataSet
    for i, i_loc in enumerate(i_locations):
        result = adjust_bias_one_location_parallel(
            obs_hist[i_loc],
            sim_hist[i_loc],
            sim_fut[i_loc],
            params,
            days,
            month_numbers,
            years)
        sim_fut_ba[i_loc] = result

    return sim_fut_ba


def adjust_bias_one_location_parallel(obs_hist_loc, sim_hist_loc, sim_fut_loc,
                                      params,
                                      days, month_numbers, years):
    """
    Bias adjusts one grid cell

    Parameters
    ----------
    obs_hist_loc: xr.DataArray
        Observational data at given location
    sim_hist_loc: xr.DataArray
        Historical simulated data at given location
    sim_fut_loc: xr.DataArray
        Future simulated data at given location
    params: Parameters
        Object that defines BA parameters
    days: dict
        Arrays of day of month for each data input
    month_numbers: dict
        Arrays of month number for each data input
    years: dict
        Arrays of year for each data input

    Returns
    -------
    sim_fut_ba_loc: xarray.DataArray
        adjusted time series with times, lat and lon
    """
    # Print out location being gridded
    # lat = float(obs_hist_loc['lat'])
    # lon = float(obs_hist_loc['lon'])
    # print(f'Gridding (lat, lon) = ({lat}, {lon})',
    #       flush=True)

    # Put in dictionary for easy iteration
    data_loc = {
        'obs_hist': obs_hist_loc,
        'sim_hist': sim_hist_loc,
        'sim_fut': sim_fut_loc
    }

    # If scaling using climatology, get upper bound for scaling
    ubc_ba = None
    if params.halfwin_ubc:
        ubcs = {
            'obs_hist': util.get_upper_bound_climatology(obs_hist_loc,
                                                         days['obs_hist'],
                                                         params.halfwin_ubc),
            'sim_hist': util.get_upper_bound_climatology(sim_hist_loc,
                                                         days['sim_hist'],
                                                         params.halfwin_ubc),
            'sim_fut': util.get_upper_bound_climatology(sim_fut_loc,
                                                        days['sim_fut'],
                                                        params.halfwin_ubc)
        }
        for key, value in data_loc.items():
            data_loc[key] = util.scale_by_upper_bound_climatology(value, ubcs[key], divide=True)

        ubc_ba = util.ccs_transfer_sim2obs_upper_bound_climatology(ubcs, days)

    # Get long term mean over each time series using valid values
    long_term_mean = {
        'obs_hist': util.average_valid_values(obs_hist_loc, np.nan,
                                              params.lower_bound, params.lower_threshold,
                                              params.upper_bound, params.upper_threshold),
        'sim_hist': util.average_valid_values(sim_hist_loc, np.nan,
                                              params.lower_bound, params.lower_threshold,
                                              params.upper_bound, params.upper_threshold),
        'sim_fut': util.average_valid_values(sim_fut_loc, np.nan,
                                             params.lower_bound, params.lower_threshold,
                                             params.upper_bound, params.upper_threshold)
    }

    # Result, to be updated as adjustment is made
    result = sim_fut_loc.copy()

    # Get window centers for running window mode
    if params.step_size:
        window_centers = util.window_centers_for_running_bias_adjustment(days, params.step_size)
        result = running_window_mode(result, window_centers, data_loc, days, years, long_term_mean, params)
    else:
        result = month_to_month_mode(result, data_loc, month_numbers, years, long_term_mean, params)

    # If we scaled variable before, time to scale back
    if params.halfwin_ubc:
        result = util.scale_by_upper_bound_climatology(result, ubc_ba, divide=False)

    # Return just resulting array if extra details not requested
    return result
