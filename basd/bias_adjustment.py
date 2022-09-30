import datetime as dt
from joblib import Parallel, delayed
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
        self.input_calendar = obs_hist.time.dt.calendar
        self.datasets = {
            'obs_hist': self.obs_hist,
            'sim_hist': self.sim_hist,
            'sim_fut': self.sim_fut
        }

        # Set dimension names to lat, lon, time
        self.set_dim_names()

        # Maps observational data onto simulated data grid resolution
        if remap_grid:
            self.obs_hist = rg.match_grids(self.obs_hist, self.sim_hist, self.sim_fut)
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
        self.sim_fut_ba = self.sim_fut

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
        days, month_numbers, years = util.time_scraping(self)

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

    def set_dim_names(self):
        """
        Makes sure latitude, longitude and time dimensions are present. These are set to be named
        lat, lon, time if not already. Will assume a matching dimension if lat, lon, or time is in
        the respective dimension name.
        """
        # For each of the datasets rename the dimensions
        for data_name, data in self.datasets.items():
            for key in data.dims:
                if 'lat' in key.lower():
                    self.datasets[data_name] = data.swap_dims({key: 'lat'})
                elif 'lon' in key.lower():
                    self.datasets[data_name] = data.swap_dims({key: 'lon'})
                elif 'time' in key.lower():
                    self.datasets[data_name] = data.swap_dims({key: 'time'})

        # Make sure each required dimension is in each dataset
        for data_name, data in self.datasets.items():
            msg = f'{data_name} needs a latitude, longitude and time dimension'
            assert all(i in data.dims for i in ['lat', 'lon', 'time']), msg

        # Save the datasets with updated dimension labels
        self.obs_hist = self.datasets['obs_hist']
        self.sim_hist = self.datasets['sim_hist']
        self.sim_fut = self.datasets['sim_fut']

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

    def abol_vec(self, i_loc):
        return self.adjust_bias_one_location(dict(lat=i_loc[0], lon=i_loc[1]))

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
        days, month_numbers, years = util.time_scraping(self)

        # Put in dictionary for easy iteration
        data_loc = {
            'obs_hist': obs_hist_loc,
            'sim_hist': sim_hist_loc,
            'sim_fut': sim_fut_loc
        }

        # If scaling using climatology, get upper bound for scaling
        ubc_ba = None
        ubcs = None
        if self.params.halfwin_ubc:
            ubcs = {
                'obs_hist': util.get_upper_bound_climatology(obs_hist_loc,
                                                             days['obs_hist'],
                                                             self.params.halfwin_ubc),
                'sim_hist': util.get_upper_bound_climatology(sim_hist_loc,
                                                             days['sim_hist'],
                                                             self.params.halfwin_ubc),
                'sim_fut': util.get_upper_bound_climatology(sim_fut_loc,
                                                            days['sim_fut'],
                                                            self.params.halfwin_ubc)
            }
            for key, value in data_loc.items():
                data_loc[key].values = util.scale_by_upper_bound_climatology(value, ubcs[key], divide=True)

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
            result.values = util.scale_by_upper_bound_climatology(result, ubc_ba, divide=False)
            sim_fut_loc.values = util.scale_by_upper_bound_climatology(sim_fut_loc, ubcs['sim_fut'], divide=False)

        # Return resulting array with extra details if requested
        if full_details:
            return olo.BaLocOutput(result, sim_fut_loc, self.variable, self.params)

        # Return just resulting array if extra details not requested
        return result

    def adjust_bias(self, lat_chunk_size: int = 0, lon_chunk_size: int = 0,
                    n_jobs: int = 1, path: str = None):
        """
        Does bias adjustment at every location of input data

        Parameters
        ----------
        n_jobs: int
            Number of jobs to request for parallelization
        path: str
            Path to save NetCDF output. If None, return result but don't create file
        lat_chunk_size: int
        lon_chunk_size: int

        Returns
        -------
        sim_fut_ba: DataSet
            Temporal grid of adjusted observations

        """
        # Get days, months and years data
        days, month_numbers, years = util.time_scraping(self)

        if lat_chunk_size & lon_chunk_size:
            # Manual chunk method
            chunk_sizes = self.obs_hist[self.variable].chunk({'lat': lat_chunk_size, 'lon': lon_chunk_size}).chunksizes

            # Indexes per chunk
            lat_indexes, lon_indexes = util.chunk_indexes(chunk_sizes)

            # Chunk to work on
            chunks = np.ndindex(len(lat_indexes), len(lon_indexes))

            chunked_results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10) \
                (delayed(adjust_bias_chunk)(
                    self.obs_hist[self.variable][dict(lat=lat_indexes[chunk[0]], lon=lon_indexes[chunk[1]])],
                    self.sim_hist[self.variable][dict(lat=lat_indexes[chunk[0]], lon=lon_indexes[chunk[1]])],
                    self.sim_fut[self.variable][dict(lat=lat_indexes[chunk[0]], lon=lon_indexes[chunk[1]])],
                    self.params,
                    self.variable,
                    days,
                    month_numbers,
                    years) for chunk in chunks)

            self.sim_fut_ba = xr.combine_by_coords(chunked_results)

        else:
            # Iterate through each location and adjust bias
            i_locations = np.ndindex(self.sizes['lat'], self.sizes['lon'])

            # Find and save results into adjusted DataSet
            results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10) \
                (delayed(adjust_bias_one_location_parallel)(
                    self.obs_hist[self.variable][dict(lat=i_loc[0], lon=i_loc[1])],
                    self.sim_hist[self.variable][dict(lat=i_loc[0], lon=i_loc[1])],
                    self.sim_fut[self.variable][dict(lat=i_loc[0], lon=i_loc[1])],
                    self.params,
                    days,
                    month_numbers,
                    years) for i_loc in i_locations)

            for i, i_loc in enumerate(i_locations):
                self.sim_fut_ba[self.variable][dict(lat=i_loc[0], lon=i_loc[1])] = results[i]

        # If provided a path to save NetCDF file, save adjusted DataSet,
        # else just return the result
        if path:
            self.save_adjustment_nc(path)
        else:
            return self.sim_fut_ba

    def save_adjustment_nc(self, path):
        """
        Saves adjusted data to NetCDF file at specific path

        Parameters
        ----------
        path: str
            Location and name of output file
        """
        self.sim_fut_ba.convert_calendar(self.input_calendar, align_on='date').to_netcdf(path)


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
                                                                      days, years,
                                                                      long_term_mean)

        # Send data to adjust bias one month
        result_this_window = util.adjust_bias_one_month(data_this_window, years_this_window,
                                                        params)

        # put central part of bias-adjusted data into result
        m_ba = util.window_indices_for_running_bias_adjustment(days['sim_fut'], window_center, 31)
        m_keep = util.window_indices_for_running_bias_adjustment(days['sim_fut'], window_center,
                                                                 params.step_size, years['sim_fut'])
        m_ba_keep = np.in1d(m_ba, m_keep)
        result.data[m_keep] = result_this_window[m_ba_keep]

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
                                                                   long_term_mean)

        # Send data to adjust bias one month
        result_this_month = util.adjust_bias_one_month(data_this_month, years_this_month, params)

        # put bias-adjusted data into result
        m = month_numbers['sim_fut'] == month
        result.data[m] = result_this_month

    return result


def adjust_bias_one_location_parallel(obs_hist_loc, sim_hist_loc, sim_fut_loc,
                                      params,
                                      days, month_numbers, years):
    """
    Bias adjusts one grid cell

    Parameters
    ----------
    obs_hist_loc: xr.DataArray
    sim_hist_loc: xr.DataArray
    sim_fut_loc: xr.DataArray
    params: Parameters
    days: dict
    month_numbers: dict
    years: dict

    Returns
    -------
    sim_fut_ba_loc: xarray.DataArray
        adjusted time series with times, lat and lon
    """
    # Print out location being gridded
    lat = float(obs_hist_loc['lat'])
    lon = float(obs_hist_loc['lon'])
    print(f'Gridding (lat, lon) = ({lat}, {lon})')

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
            data_loc[key].values = util.scale_by_upper_bound_climatology(value, ubcs[key], divide=True)

        ubc_ba = util.ccs_transfer_sim2obs_upper_bound_climatology(ubcs, days)

    # Get long term mean over each time series using valid values
    long_term_mean = {
        'obs_hist': util.average_valid_values(obs_hist_loc.values, np.nan,
                                              params.lower_bound, params.lower_threshold,
                                              params.upper_bound, params.upper_threshold),
        'sim_hist': util.average_valid_values(sim_hist_loc.values, np.nan,
                                              params.lower_bound, params.lower_threshold,
                                              params.upper_bound, params.upper_threshold),
        'sim_fut': util.average_valid_values(sim_fut_loc.values, np.nan,
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
        result.values = util.scale_by_upper_bound_climatology(result, ubc_ba, divide=False)

    # Return just resulting array if extra details not requested
    return result


def adjust_bias_chunk(obs_hist, sim_hist, sim_fut, params, variable, days, month_numbers, years):
    # Iterate through each location and adjust bias
    i_locations = np.ndindex(obs_hist.sizes['lat'], obs_hist.sizes['lon'])

    # Unadjusted results of correct shape
    sim_fut_ba = xr.Dataset({variable: sim_fut})

    # Find and save results into adjusted DataSet
    for i, i_loc in enumerate(i_locations):
        result = adjust_bias_one_location_parallel(
            obs_hist[dict(lat=i_loc[0], lon=i_loc[1])],
            sim_hist[dict(lat=i_loc[0], lon=i_loc[1])],
            sim_fut[dict(lat=i_loc[0], lon=i_loc[1])],
            params,
            days,
            month_numbers,
            years)
        sim_fut_ba[variable][dict(lat=i_loc[0], lon=i_loc[1])] = result

    return sim_fut_ba
