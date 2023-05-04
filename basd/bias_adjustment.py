import datetime as dt
import os
import shutil

from dask.distributed import progress
import dask.array as da
import numpy as np
import xarray as xr

from basd.ba_params import Parameters
import basd.regridding as rg
import basd.utils as util
import basd.one_loc_output as olo


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


def init_bias_adjustment(obs_hist: xr.Dataset,
                         sim_hist: xr.Dataset,
                         sim_fut: xr.Dataset,
                         variable: str,
                         params: Parameters,
                         remap_grid: bool = False,
                         temp_path: str = 'basd_temp_path',
                         time_chunk: int = 100,
                         periodic: bool = True):
    """
    Parameters
    ----------
    obs_hist: xr.Dataset
        Historical observation data grid
    sim_hist: xr.Dataset
        Historical simulation data grid
    sim_fut: xr.Dataset
        Future simulation data grid
    variable: str
        Name of the variable of interest
    params: Parameters
        Object that specifies the parameters for the variable's SD routine
    temp_path: str
        path to directory where intermediate files are stored
    time_chunk: int
        size of chunks in time dimension before writing to zarr. Should be large for speed, but just small enough to fit into memory if that's an issue. Otherwise not important.
    periodic: bool
        Whether grid wraps around globe longitudinally. Used during regridding interpolation.

    Returns
    -------
    init_output: dict
        Dictionary of details that need to be passed along into the bias adjustment process
    """
    # Setting the data
    # Also converting calendar to proleptic_gregorian.
    # See xarray.convert_calendar() fro details, but default uses date alignment
    # for 360_day calendars, as this preserves month data for month-to-month mode
    obs_hist = obs_hist.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
    sim_hist = sim_hist.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
    sim_fut = sim_fut.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
    
    # Saving input calendar type to convert back at end
    input_calendar = sim_hist.time.dt.calendar

    # Dictionary of datasets for iteration
    datasets = {
        'obs_hist': obs_hist,
        'sim_hist': sim_hist,
        'sim_fut': sim_fut
    }

    # Set dimension names to lat, lon, time
    datasets = util.set_dim_names(datasets)
    obs_hist = datasets['obs_hist']
    sim_hist = datasets['sim_hist']
    sim_fut = datasets['sim_fut']

    # Maps observational data onto simulated data grid resolution
    if remap_grid:
        obs_hist = rg.project_onto(obs_hist, sim_hist, variable, periodic)
        datasets['obs_hist'] = obs_hist

    # Forces data to have same spatial shape and resolution
    obs_hist, sim_hist, sim_fut, datasets = assert_consistency_of_data_resolution(datasets)

    # Assert full period coverage 
    obs_hist, sim_hist, sim_fut = assert_full_period_coverage(datasets)

    # Get time information
    days, month_numbers, years = util.time_scraping(datasets)

    # Save intermediate arrays
    obs_hist_write = obs_hist.chunk(dict(lon=obs_hist.dims['lon'], lat=obs_hist.dims['lat'], time=time_chunk)).\
        to_zarr(os.path.join(temp_path, 'obs_hist_init.zarr'), mode='w')
    progress(obs_hist_write)
    sim_hist_write = sim_hist.chunk(dict(lon=sim_hist.dims['lon'], lat=sim_hist.dims['lat'], time=time_chunk)).\
        to_zarr(os.path.join(temp_path, 'sim_hist_init.zarr'), mode='w')
    progress(sim_hist_write)
    sim_fut_write = sim_fut.chunk(dict(lon=sim_fut.dims['lon'], lat=sim_fut.dims['lat'], time=time_chunk)).\
        to_zarr(os.path.join(temp_path, 'sim_fut_init.zarr'), mode='w')
    progress(sim_fut_write)

    obs_hist.close()
    sim_hist.close()
    sim_fut.close()

    # Return info that needs to be tracked
    return {
        'temp_path': temp_path,
        'variable': variable,
        'params': params,
        'input_calendar': input_calendar, 
        'days': days, 
        'month_numbers': month_numbers, 
        'years': years
        }


def assert_full_period_coverage(datasets):
    """
    Raises an assertion error if years aren't fully covered. Trims data to the first Jan 1st
    available to the last Dec 31st available.
    """

    for key, data in datasets.items():
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
        datasets[key] = data.sel(time=keep)

    # Updating data
    obs_hist = datasets['obs_hist']
    sim_hist = datasets['sim_hist']
    sim_fut = datasets['sim_fut']

    # Getting updated time info
    days, month_numbers, years = util.time_scraping(datasets)

    # Asserting all years are here within range of first and last
    for key, data in datasets.items():
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

        return obs_hist, sim_hist, sim_fut


def assert_consistency_of_data_resolution(datasets):
    """
    Raises an assertion error if data are not of the same shape or cannot be aggregated to
    the same resolution. Otherwise, it will force data to have same spatial resolution via
    aggregation.
    """
    obs_hist = datasets['obs_hist']
    sim_hist = datasets['sim_hist']
    sim_fut = datasets['sim_fut']

    # Get the most coarse dimensions
    min_lat = min([obs_hist.sizes.get('lat'), sim_hist.sizes.get('lat'), sim_fut.sizes.get('lat')])
    min_lon = min([obs_hist.sizes.get('lon'), sim_hist.sizes.get('lon'), sim_fut.sizes.get('lon')])

    # For each dataset, aggregate to the most coarse dataset if possible
    for key, value in datasets.items():
        agg_lat = value.sizes.get('lat') / min_lat
        agg_lon = value.sizes.get('lon') / min_lon
        assert agg_lat == agg_lon, f'Data have differing shapes'
        assert agg_lat.is_integer(), f'Non-integer aggregation factor for {key}'
        agg_fact = int(agg_lat)
        if agg_fact > 1:
            print(f'Aggregating {key} by a factor of {agg_fact}')
            datasets[key] = value.coarsen(lat=agg_fact).mean().coarsen(lon=agg_fact).mean()

    # Save aggregated datasets
    obs_hist = datasets['obs_hist']
    sim_hist = datasets['sim_hist']
    sim_fut = datasets['sim_fut']

    return obs_hist, sim_hist, sim_fut, datasets


def adjust_bias(init_output, clear_temp: bool = True, lat_chunk_size: int = 0, lon_chunk_size: int = 0,
                file: str = None, monthly: bool = False, encoding = None):
    """
    Does bias adjustment at every location of input data

    Parameters
    ----------
    init_output: dict
        Dictionary of details relevant to Bias Adjustment, created during initialization
    clear_temp: bool
        Whether or not to clear temporary directory when done. Cleared by default.
    lat_chunk_size: int
        Number of cells to include in chunk in lat direction
    lon_chunk_size: int
        Number of cells to include in chunk in lon direction
    file: str
        Where to save the output as .nc file. (Optional, can just return output without saving)
    encoding: dict
        Parameter for to_netcdf function
    monthly: bool
        Whether to aggregate to monthly data before saving

    Returns
    -------
    sim_fut_ba: DataSet
        Temporal grid of adjusted observations

    """
    # Get details from initialization that we use multiple times
    temp_path = init_output['temp_path']
    variable = init_output['variable']
    params = init_output['params']
    days = init_output['days']
    month_numbers = init_output['month_numbers']
    years = init_output['years']

    # Read in data
    obs_hist = xr.open_zarr(os.path.join(temp_path, 'obs_hist_init.zarr'))
    sim_hist = xr.open_zarr(os.path.join(temp_path, 'sim_hist_init.zarr'))
    sim_fut = xr.open_zarr(os.path.join(temp_path, 'sim_fut_init.zarr'))

    # Order dimensions lon, lat, time
    obs_hist[variable] = obs_hist[variable].transpose('lon', 'lat', 'time')
    sim_hist[variable] = sim_hist[variable].transpose('lon', 'lat', 'time')
    sim_fut[variable] = sim_fut[variable].transpose('lon', 'lat', 'time')

    # Chunk data
    if lat_chunk_size & lon_chunk_size:
        # Manual chunk method
        obs_hist = obs_hist.chunk(dict(lon=lon_chunk_size, lat=lat_chunk_size, time=-1))
        sim_hist = sim_hist.chunk(dict(lon=lon_chunk_size, lat=lat_chunk_size, time=-1))
        sim_fut = sim_fut.chunk(dict(lon=lon_chunk_size, lat=lat_chunk_size, time=-1))

    else:
        # Auto chunk method (not allowing to be chunked over time)
        obs_hist = obs_hist.chunk(dict(lon=None, lat=None, time=-1))
        sim_hist = sim_hist.chunk(dict(lon=None, lat=None, time=-1))
        sim_fut = sim_fut.chunk(dict(lon=None, lat=None, time=-1))

    # Force wait till this step is done before continuing
    progress(obs_hist)
    progress(sim_hist)
    progress(sim_fut)

    # Make temp copy of correct shape for final data
    sim_fut_ba = sim_fut.copy()

    # Set up dask computation
    ba_output_data = da.map_blocks(adjust_bias_chunk,
                                    obs_hist[variable].data,
                                    sim_hist[variable].data,
                                    sim_fut[variable].data,
                                    params=params,
                                    days=days, month_numbers=month_numbers, years=years,
                                    dtype=object, chunks=sim_fut[variable].chunks)

    # Save output
    sim_fut_ba[variable].data = ba_output_data

    # If provided a path to save NetCDF file, save adjusted DataSet,
    # else just return the result
    if file:
        save_adjustment_nc(sim_fut_ba, file, init_output['input_calendar'], variable, monthly, encoding)

    # Clear the temporary directory. Optional but happens by default
    if clear_temp:
        try:
            shutil.rmtree(init_output['temp_path'])
        except OSError as e:
            print("Error: %s : %s" % (init_output['temp_path'], e.strerror))
    
    return sim_fut_ba


def save_adjustment_nc(sim_fut_ba, file, input_calendar, variable, monthly: bool = False, encoding=None):
    """
    Saves adjusted data to NetCDF file at specific path

    Parameters
    ----------
    file: str
        Location and name string to save output file
    encoding: dict
        Parameter for to_netcdf function
    monthly: bool
        Whether to aggregate to monthly data before saving
    """

    # Make sure we've computed
    sim_fut_ba = sim_fut_ba.persist()

    # If monthly, save monthly aggregation
    if monthly:
        temp = sim_fut_ba.astype(float).\
            resample(time='1MS').\
            mean(dim='time').\
            chunk({'time': -1}).\
            copy()
        write_job = temp.to_netcdf(file, encoding={variable: encoding}, compute=True)
        progress(write_job)
    else:
        # Try converting calendar back to input calendar
        try:
            sim_fut_ba = sim_fut_ba.convert_calendar(input_calendar, align_on='date')
        except AttributeError:
            AttributeError('Unable to convert calendar')

        write_job = sim_fut_ba.to_netcdf(file, encoding={variable: encoding}, compute=True)
        progress(write_job)


def adjust_bias_one_location(init_output, i_loc, clear_temp: bool = False, full_details: bool = True):
    """
    Bias adjusts one grid cell

    Parameters
    ----------
    init_output: dict
        Dictionary of details relevant to Bias Adjustment, created during initialization
    clear_temp: bool
        Whether or not to clear temporary directory when done. Left alone by default.
    i_loc: dict
        index of grid cell to bias adjust
    full_details: bool
        Should function return full details of run, or just the time series array

    Returns
    -------
    sim_fut_ba_loc: xarray.DataArray
        adjusted time series with times, lat and lon
    """
    # Get details from initialization that we use multiple times
    temp_path = init_output['temp_path']
    variable = init_output['variable']
    params = init_output['params']
    days = init_output['days']
    month_numbers = init_output['month_numbers']
    years = init_output['years']

    # Read in data
    obs_hist = xr.open_zarr(os.path.join(temp_path, 'obs_hist_init.zarr'))
    sim_hist = xr.open_zarr(os.path.join(temp_path, 'sim_hist_init.zarr'))
    sim_fut = xr.open_zarr(os.path.join(temp_path, 'sim_fut_init.zarr'))

    # Get data at one location
    obs_hist_loc = obs_hist[variable][i_loc]
    sim_hist_loc = sim_hist[variable][i_loc]
    sim_fut_loc = sim_fut[variable][i_loc]

    # Put in dictionary for easy iteration
    data_loc = {
        'obs_hist': obs_hist_loc.values,
        'sim_hist': sim_hist_loc.values,
        'sim_fut': sim_fut_loc.values
    }

    # If scaling using climatology, get upper bound for scaling
    ubc_ba = None
    ubcs = None
    if params.halfwin_ubc:
        ubcs = {
            'obs_hist': util.get_upper_bound_climatology(obs_hist_loc.values,
                                                            days['obs_hist'],
                                                            params.halfwin_ubc),
            'sim_hist': util.get_upper_bound_climatology(sim_hist_loc.values,
                                                            days['sim_hist'],
                                                            params.halfwin_ubc),
            'sim_fut': util.get_upper_bound_climatology(sim_fut_loc.values,
                                                        days['sim_fut'],
                                                        params.halfwin_ubc)
        }
        for key, value in data_loc.items():
            data_loc[key].values = util.scale_by_upper_bound_climatology(value.values, ubcs[key], divide=True)

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
        return olo.BaLocOutput(result, sim_fut_loc, obs_hist_loc, sim_hist_loc, variable, params)

    # Return just resulting array if extra details not requested
    return result
