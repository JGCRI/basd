import os
import shutil
import warnings

import dask.array as da
from dask.distributed import progress
import numpy as np
import scipy.linalg as spl
import xarray as xr

from basd.ba_params import Parameters
import basd.regridding as rg
import basd.sd_utils as sdu
import basd.utils as util


def get_data_at_loc(loc,
                    obs_fine, sim_coarse, sim_fine,
                    month_numbers, downscaling_factors, sum_weights):
    """
    Function for extracting the data from the desired location and in the desired form

    Parameters
    ----------
    loc: tuple
        (lon index, lat index)
    obs_fine
    sim_coarse
    sim_fine
    month_numbers
    downscaling_factors
    sum_weights

    Returns
    -------
    data: dict
    sum_weights_loc: array
    """
    # Empty data dict
    data = {}

    # Values of the coarse cell over time period
    coarse_values = sim_coarse[loc].reshape(month_numbers['sim_coarse'].size)
    data['sim_coarse'] = coarse_values

    # Range of indexes of fine cells
    lat_start = loc[0] * downscaling_factors['lat']
    lat_end = (loc[0] + 1) * downscaling_factors['lat']
    lon_start = loc[1] * downscaling_factors['lon']
    lon_end = (loc[1] + 1) * downscaling_factors['lon']

    # Get the cell weights of relevant latitudes
    sum_weights_loc = sum_weights[lat_start:lat_end, lon_start:lon_end]

    # Value of fine observational cells
    obs_fine_values = obs_fine[lat_start:lat_end, lon_start:lon_end]

    # Values of fine simulated cells
    sim_fine_values = sim_fine[lat_start:lat_end, lon_start:lon_end]

    # reshape. (lat, lon, time) --> (time, lat x lon) ex.) [4 x 4 x 365] --> [365 x 16]
    # Spreads fine cell at single time point, first along row then cols.
    #  1  2  3  4     \    time 1:  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    #  5  6  7  8  --- \   time 2:  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    #  9 10 11 12  --- /   time 3:  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    # 13 14 15 16     /    time 4:  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    data['obs_fine'] = obs_fine_values.reshape((downscaling_factors['lat'] * downscaling_factors['lon'],
                                                month_numbers['obs_fine'].size)).T
    data['sim_fine'] = sim_fine_values.reshape((downscaling_factors['lat'] * downscaling_factors['lon'],
                                                month_numbers['sim_coarse'].size)).T
    sum_weights_loc = sum_weights_loc.reshape(downscaling_factors['lat'] * downscaling_factors['lon'])

    return data, sum_weights_loc


def downscale_chunk(obs_fine, sim_coarse, sim_fine, weights,
                    params, month_numbers,
                    downscaling_factors, rotation_matrices):
    """
    Performs the downscaling routine on each cell of the provided chunk

    Parameters
    ----------
    obs_fine: da.Array
        (M,N) observational grid data for a given lat x lon chunk
    sim_coarse: da.Array
        (m,n) simulated grid data for a given lat x lon chunk
    sim_fine: da.Array
        (M,N) simulated grid data for a given lat x lon chunk
    params: Parameters
        Object that defines parameters for variable at hand
    weights: da.Array
        Weights of fine grids cells according to relative global area
    month_numbers: dict
        Arrays that relate data to the month number it's associated with
    downscaling_factors: dict
        M/n and N/n, factor between the two different grid resolutions
    rotation_matrices: list
        (N,N) ndarray orthogonal rotation matrices

    Returns
    -------
    output_chunk: da.Array
        (M,N) simulated grid data for a given lat x lon chunk
    """
    # Copy of fine resolution simulated data to fill with results
    output_chunk = sim_fine.copy()

    # Iterable for each cell in chunk
    i_locations = np.ndindex(sim_coarse.shape[0], sim_coarse.shape[1])

    # Iterate through each cell in coarse chunk
    for i_loc in i_locations:
        # Get the climate data and grid cell areas at given location
        data_weights = get_data_at_loc(i_loc,
                                       obs_fine, sim_coarse, sim_fine,
                                       month_numbers, downscaling_factors, weights)

        # Performing the downscaling algorithm
        result = downscale_one_location_parallel(i_loc, params, data_weights,
                                                 month_numbers, downscaling_factors,
                                                 rotation_matrices)

        # Save the result to the output chunk object
        lat_start = i_loc[0] * downscaling_factors['lat']
        lat_end = (i_loc[0] + 1) * downscaling_factors['lat']
        lon_start = i_loc[1] * downscaling_factors['lon']
        lon_end = (i_loc[1] + 1) * downscaling_factors['lon']
        output_chunk[lat_start:lat_end, lon_start:lon_end] = result

    # Return updated chunk
    return output_chunk


def downscale_one_location_parallel(loc, params,
                                    data_weights,
                                    month_numbers, downscaling_factors,
                                    rotation_matrices):
    """
    Downscales a single coarse grid cell to the resolution of the fine observational grid,
    at the given location.

    Parameters
    ----------
    rotation_matrices
    params
    data_weights
    downscaling_factors
    loc
    month_numbers

    Returns
    -------
    sim_fine_loc: ndarray
        3D array of downscaled coarse grid. Size is latitude scaling factor by longitude scaling
        factor by time
    """

    # Get data at location
    data, sum_weights_loc = data_weights

    # abort here if there are only missing values in at least one time series
    # do not abort though if the if_all_invalid_use option has been specified
    if np.isnan(params.if_all_invalid_use):
        if sdu.only_missing_values_in_at_least_one_time_series(data):
            warnings.warn(f'{loc} skipped due to missing data')
            return None

    # compute mean value over all time steps for invalid value sampling
    long_term_mean = {}
    for key, d in data.items():
        long_term_mean[key] = util.average_valid_values(d, params.if_all_invalid_use,
                                                        params.lower_bound, params.lower_threshold,
                                                        params.upper_bound, params.upper_threshold)

    # Result in flattened format
    result = downscale_month_by_month(data, sum_weights_loc,
                                      long_term_mean, month_numbers,
                                      rotation_matrices, params)

    # Reshape to grid
    result = result.T.reshape(downscaling_factors['lat'],
                              downscaling_factors['lon'],
                              month_numbers['sim_fine'].size)

    return result


def downscale_month_by_month(data, sum_weights, long_term_mean, month_numbers, rotation_matrices, params):
    # Get form of result
    result = data['sim_fine'].copy()

    # Get data this month
    data_this_month = {}
    for month in params.months:
        # extract data
        for key, d in data.items():
            m = month_numbers[key] == month
            assert np.any(m), f'No data found for month {month}, in {key}'
            valid_values, invalid_values = util.sample_invalid_values(d[m],
                                                                      params.randomization_seed,
                                                                      long_term_mean[key])
            # TODO: What is up with the parameters for randomize_censored_values?
            randomized_censored_values = util.randomize_censored_values(valid_values,
                                                                        params.lower_bound, params.lower_threshold,
                                                                        params.upper_bound, params.upper_threshold,
                                                                        False, False,
                                                                        params.randomization_seed,
                                                                        10., 10.)

            data_this_month[key] = randomized_censored_values

        # do statistical downscaling
        result_this_month = downscale_one_month(data_this_month, rotation_matrices,
                                                params.lower_bound, params.lower_threshold,
                                                params.upper_bound, params.upper_threshold,
                                                sum_weights, params)

        # put downscaled data into result
        m = month_numbers['sim_fine'] == month
        result[m] = result_this_month

    return result


def downscale_one_month(data_this_month, rotation_matrices,
                        lower_bound, lower_threshold,
                        upper_bound, upper_threshold,
                        sum_weights, params):
    """
    Applies the MBCn algorithm with weight preservation. Randomizes censored values

    Parameters
    ----------
    data_this_month: dict
        Keys: obs_fine, sim_coarse, sim_fine
        obs_fine: ndarray, observational grid data in the given month
        sim_coarse: array, simulated coarse data. One point per time. All values in given month
        sim_fine: ndarray, sim_coarse but interpolated to obs_fine spatial resolution
    rotation_matrices: list
        (N,N) ndarray orthogonal rotation matrices
    lower_bound: float/None
        Lower bound of variable at hand
    lower_threshold: float
        Lower threshold of variable at hand
    upper_bound: float/None
        Upper bound of variable at hand
    upper_threshold: float/None
        Upper threshold of variable at hand
    sum_weights: array
        Weights of fine grids cells according to relative global area
    params: Parameters
        Object that defines parameters for variable at hand

    Returns
    -------
    sim_fine: ndarray
        grid of statistically downscaled values
    """

    mbcn_sd_result = sdu.weighted_sum_preserving_mbcn(data_this_month['obs_fine'],
                                                      data_this_month['sim_coarse'],
                                                      data_this_month['sim_fine'],
                                                      sum_weights,
                                                      rotation_matrices,
                                                      params.n_quantiles)

    sim_fine = util.randomize_censored_values(mbcn_sd_result,
                                              lower_bound, lower_threshold,
                                              upper_bound, upper_threshold,
                                              False, True)
    # TODO: assert no infs or nans

    return sim_fine


def generate_cre_matrix(n: int):
    """
    Parameters
    ----------
    n: int
        Number of rows and columns of the CRE matrix

    Returns
    -------
    mat: (n,n) ndarray
        CRE matrix.
    """
    z = np.random.randn(n, n)
    q, r = spl.qr(z)  # QR decomposition
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def init_downscaling(obs_fine: xr.Dataset,
                     sim_coarse: xr.Dataset,
                     variable: str,
                     params: Parameters,
                     temp_path: str = 'basd_temp_path',
                     time_chunk: int = 100,
                     periodic: bool = True):
    """
    Parameters
    ----------
    obs_fine: xr.Dataset
        Fine grid of observational data
    sim_coarse: xr.Dataset
        Coarse grid of simulated data
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
        Dictionary of details that need to be passed along into the downscaling process
    """

    # Set base input data
    obs_fine = obs_fine.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
    sim_coarse = sim_coarse.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)

    # Input calendar of the simulation model
    input_calendar = sim_coarse.time.dt.calendar

    # Dictionary of datasets to iterate through easily
    datasets = {
        'obs_fine': obs_fine,
        'sim_coarse': sim_coarse,
        'sim_fine': sim_coarse.copy()
    }

    # Set dimension names to lat, lon, time
    datasets = util.set_dim_names(datasets)
    obs_fine = datasets['obs_fine']
    sim_coarse = datasets['sim_coarse']

    # Get time details
    days, month_numbers, years = util.time_scraping(datasets)
    del datasets

    # Interpolate coarse grid to be conforming with fine grid
    sim_coarse = rg.reproject_for_integer_factors(obs_fine, sim_coarse, variable, periodic)

    # Analyze input grids
    downscaling_factors = analyze_input_grids(obs_fine, sim_coarse)

    # Set downscaled grid as copy of fine grid for now
    sim_fine: xr.Dataset = rg.project_onto(sim_coarse, obs_fine, variable, periodic)

    # Grid cell weights by global area
    sum_weights = grid_cell_weights(obs_fine.coords['lat'].values)

    # get list of rotation matrices to be used for all locations and months
    if params.randomization_seed is not None:
        np.random.seed(params.randomization_seed)
    rotation_matrices = [generate_cre_matrix(downscaling_factors['lat'] * downscaling_factors['lon'])
                            for _ in range(params.n_iterations)]

    # Save intermediate arrays
    obs_fine_write = obs_fine.chunk(dict(lon=obs_fine.dims['lon'], lat=obs_fine.dims['lat'], time=time_chunk)).\
        to_zarr(os.path.join(temp_path, 'obs_fine_init.zarr'), mode='w')
    progress(obs_fine_write)
    sim_fine_write = sim_fine.chunk(dict(lon=sim_fine.dims['lon'], lat=sim_fine.dims['lat'], time=time_chunk)).\
        to_zarr(os.path.join(temp_path, 'sim_fine_init.zarr'), mode='w')
    progress(sim_fine_write)
    sim_coarse_write = sim_coarse.chunk(dict(lon=sim_coarse.dims['lon'], lat=sim_coarse.dims['lat'], time=time_chunk)).\
        to_zarr(os.path.join(temp_path, 'sim_coarse_init.zarr'), mode='w')
    progress(sim_coarse_write)

    obs_fine.close()
    sim_fine.close()
    sim_coarse.close()

    # Return info that needs to be tracked
    return {
        'temp_path': temp_path,
        'variable': variable,
        'params': params,
        'downscaling_factors': downscaling_factors,
        'input_calendar': input_calendar, 
        'sum_weights': sum_weights, 
        'rotation_matrices': rotation_matrices, 
        'days': days, 
        'month_numbers': month_numbers, 
        'years': years
        }


def analyze_input_grids(obs_fine, sim_coarse):
    """
    Asserts that grids are of compatible sizes, and returns scaling factors,
    direction of coordinate sequence, and whether the sequence is circular

    Returns
    -------
    downscaling_factors: dict
    """
    # Coordinate sequences
    fine_lats = obs_fine.coords['lat'].values
    fine_lons = obs_fine.coords['lon'].values
    coarse_lats = sim_coarse.coords['lat'].values
    coarse_lons = sim_coarse.coords['lon'].values

    # Assert that fine grid is a multiple of the coarse
    assert len(fine_lats) % len(coarse_lats) == 0
    assert len(fine_lons) % len(coarse_lons) == 0

    # Get the downscaling factors
    f_lat = len(fine_lats) // len(coarse_lats)
    f_lon = len(fine_lons) // len(coarse_lons)

    # Assert that we really are trying to downscale (not stay the same or upscale)
    assert f_lat > 1 or f_lon > 1, f'No downscaling needed. Observational grid provides no finer resolution'

    # Step sizes in coordinate sequences
    coarse_lat_deltas = np.diff(coarse_lats)
    fine_lat_deltas = np.diff(fine_lats)
    coarse_lon_deltas = np.diff(coarse_lons)
    fine_lon_deltas = np.diff(fine_lons)

    # Assert that the sequences move in the same direction and are monotone
    assert (np.all(coarse_lat_deltas > 0) and np.all(fine_lat_deltas > 0)) or \
            (np.all(coarse_lat_deltas < 0) and np.all(fine_lat_deltas < 0)), f'Latitude coords should be ' \
                                                                            f'monotonic in the same direction.'
    assert (np.all(coarse_lon_deltas > 0) and np.all(fine_lon_deltas > 0)) or \
            (np.all(coarse_lon_deltas < 0) and np.all(fine_lon_deltas < 0)), f'Longitude coords should be ' \
                                                                            f'monotonic in the same direction.'

    # Assert a constant delta in sequences
    assert np.allclose(coarse_lat_deltas, coarse_lat_deltas[0]) and np.allclose(
        fine_lat_deltas, fine_lat_deltas[0]), f'Resolution should be constant for all latitude'
    assert np.allclose(coarse_lon_deltas, coarse_lon_deltas[0]) and np.allclose(
        fine_lon_deltas, fine_lon_deltas[0]), f'Resolution should be constant for all longitude'

    # Save the scaling factors
    scale_factors = {
        'lat': f_lat,
        'lon': f_lon
    }

    return scale_factors


def grid_cell_weights(lats):
    """
    Function for finding the weight of each grid cell based on global grid area

    Parameters
    ----------
    lats: np.array
        numpy array of latitudes of fine grid

    Returns
    -------
    weight_by_lat: np.array
        numpy array of grid cell area by latitude for cells in fine grid
    """
    weight_by_lat = np.cos(np.deg2rad(lats))

    return weight_by_lat


def downscale(init_output, output_dir: str = None, clear_temp: bool = True,
              lat_chunk_size: int = 0, lon_chunk_size: int = 0,
              day_file: str = None, month_file: str = None, encoding = None):
    """
    Function to downscale climate data using MBCn_SD method

    Parameters
    ----------
    file: str
        Location and name string to save output file
    lat_chunk_size: int
        Number of cells to include in chunk in lat direction
    lon_chunk_size: int
        Number of cells to include in chunk in lon direction
    encoding: dict
        Parameter for save as netcdf function

    Returns
    -------
    sim_fine: xr.Dataset
        Downscaled data. Same spatial resolution as input obs_fine
    """
    # Get corresponding chunks for coarse data
    fine_lon_chunk_size = lon_chunk_size * init_output['downscaling_factors']['lon']
    fine_lat_chunk_size = lat_chunk_size * init_output['downscaling_factors']['lat']

    # Open data
    obs_fine = xr.open_zarr(os.path.join(init_output['temp_path'], 'obs_fine_init.zarr'))
    sim_fine = xr.open_zarr(os.path.join(init_output['temp_path'], 'sim_fine_init.zarr'))
    sim_coarse = xr.open_zarr(os.path.join(init_output['temp_path'], 'sim_coarse_init.zarr'))

    # Order dimensions lon, lat, time
    obs_fine[init_output['variable']] = obs_fine[init_output['variable']].transpose('lat', 'lon', 'time')
    sim_fine[init_output['variable']] = sim_fine[init_output['variable']].transpose('lat', 'lon', 'time')
    sim_coarse[init_output['variable']] = sim_coarse[init_output['variable']].transpose('lat', 'lon', 'time')

    # Chunk data
    obs_fine = obs_fine.chunk(dict(lat=fine_lat_chunk_size, lon=fine_lon_chunk_size, time=-1))
    progress(obs_fine) 
    sim_fine = sim_fine.chunk(dict(lat=fine_lat_chunk_size, lon=fine_lon_chunk_size, time=-1))
    progress(sim_fine) 
    sim_coarse = sim_coarse.chunk(dict(lat=lat_chunk_size, lon=lon_chunk_size, time=-1))
    progress(sim_coarse) 
    sim_fine_out = sim_fine.copy()

    # Chunk grid area cell weights
    fine_size = tuple((obs_fine.sizes['lat'], obs_fine.sizes['lon'], 1))
    sum_weights = init_output['sum_weights'].repeat(fine_size[1]).reshape(fine_size)
    fine_chunk_size = tuple((fine_lat_chunk_size, fine_lon_chunk_size, 1))
    chunk_sum_weights = da.from_array(sum_weights, chunks=fine_chunk_size)

    # Downscale with dask map_blocks handling parallelization
    # Set up dask computation
    ba_output_data = da.map_blocks(downscale_chunk,
                                    obs_fine[init_output['variable']].data,
                                    sim_coarse[init_output['variable']].data,
                                    sim_fine[init_output['variable']].data,
                                    chunk_sum_weights,
                                    params=init_output['params'],
                                    month_numbers=init_output['month_numbers'],
                                    downscaling_factors=init_output['downscaling_factors'],
                                    rotation_matrices=init_output['rotation_matrices'],
                                    dtype=object, chunks=sim_fine[init_output['variable']].chunks)

    # Put statistically downscaled data into the new Dataset
    sim_fine_out[init_output['variable']].data = ba_output_data

    # If an output file is provided, write data to that file as .nc
    if day_file or month_file:
        save_downscale_nc(sim_fine_out, init_output['variable'], init_output['input_calendar'], output_dir, day_file, month_file, encoding)

    # Clear the temporary directory. Optional but happens by default
    if clear_temp:
        try:
            shutil.rmtree(init_output['temp_path'])
        except OSError as e:
            print("Error: %s : %s" % (init_output['temp_path'], e.strerror))

    return sim_fine_out


def save_downscale_nc(sim_fine_out, variable, input_calendar, output_dir, day_file: str = None, month_file: str = None, encoding = None):
    """
    Saves Downscaled data to NetCDF files at specific path

    Parameters
    ----------
    file: str
        Location to and name of file to save downscaled data
    encoding: dict
        Parameter for to_netcdf function
    """
    # Make sure we've computed
    sim_fine_out = sim_fine_out.persist()

    # If not saving daily data in long term
    day_flag = False
    if not day_file:
        day_flag = True
        day_file = 'sim_fut_ba_day.nc'

    # Try converting calendar back to input calendar
    try:
        sim_fine_out = sim_fine_out.convert_calendar(input_calendar, align_on='date')
    except AttributeError:
        AttributeError('Unable to convert calendar')

    # Save daily data
    write_job = sim_fine_out.to_netcdf(os.path.join(output_dir, day_file), encoding={variable: encoding}, compute=True)
    progress(write_job)
    sim_fine_out.close()

    # If monthly, save monthly aggregation
    if month_file:
        sim_fine_out = xr.open_mfdataset(os.path.join(output_dir, day_file), chunks={'time': 50})
        temp = sim_fine_out.astype(float).\
            resample(time='1MS').\
            mean(dim='time').\
            chunk({'time': -1}).\
            copy()
        write_job = temp.to_netcdf(os.path.join(output_dir, month_file), encoding={variable: encoding}, compute=True)
        progress(write_job)
        temp.close()
        sim_fine_out.close()
    
    # Delete daily data if not wanted
    if day_flag:
        os.remove(os.path.join(output_dir, day_file))


def downscale_one_location(init_output, i_loc: dict, clear_temp: bool = False):
    """
    Function to downscale a single coarse grid cell

    Parameters
    ----------
    init_output: dict
        output of the downscaling initialization
    i_loc: dict
        Dictionary of lat, lon index, for coarse cell
    clear_temp: bool
        Whether to clear intermediate results after downscaling at this location. Default is not to do this.

    Returns
    -------
    sim_fine_loc: ndarray
        3D array of downscaled coarse grid. Size is latitude scaling factor by longitude scaling
        factor by time
    """
    # Extract intitialization details that we use multiple times
    temp_path = init_output['temp_path']
    variable = init_output['variable']
    params = init_output['params']
    month_numbers = init_output['month_numbers']
    downscaling_factors = init_output['downscaling_factors']

    # Open data
    obs_fine = xr.open_zarr(os.path.join(temp_path, 'obs_fine_init.zarr'))
    sim_fine = xr.open_zarr(os.path.join(temp_path, 'sim_fine_init.zarr'))
    sim_coarse = xr.open_zarr(os.path.join(temp_path, 'sim_coarse_init.zarr'))

    # Extract intitialization details that we use multiple times
    variable = init_output['variable']
    params = init_output['params']
    month_numbers = init_output['month_numbers']
    downscaling_factors = init_output['downscaling_factors']

    # Order dimensions lon, lat, time
    obs_fine[variable] = obs_fine[variable].transpose('lat', 'lon', 'time')
    sim_fine[variable] = sim_fine[variable].transpose('lat', 'lon', 'time')
    sim_coarse[variable] = sim_coarse[variable].transpose('lat', 'lon', 'time')

    # Turn location dictionary into tuple (lat index, lon index)
    loc_tuple = (i_loc['lat'], i_loc['lon'])

    # Get data at location
    data, sum_weights_loc = get_data_at_loc(loc_tuple,
                                            obs_fine[variable].data,
                                            sim_coarse[variable].data,
                                            sim_fine[variable].data,
                                            month_numbers, downscaling_factors, init_output['sum_weights'])

    # abort here if there are only missing values in at least one time series
    # do not abort though if the if_all_invalid_use option has been specified
    if np.isnan(params.if_all_invalid_use):
        if sdu.only_missing_values_in_at_least_one_time_series(data):
            warnings.warn(f'{i_loc} skipped due to missing data')
            return None

    # compute mean value over all time steps for invalid value sampling
    long_term_mean = {}
    for key, d in data.items():
        long_term_mean[key] = util.average_valid_values(d, params.if_all_invalid_use,
                                                        params.lower_bound, params.lower_threshold,
                                                        params.upper_bound, params.upper_threshold)

    # Result in flattened format
    result = downscale_month_by_month(data, sum_weights_loc,
                                        long_term_mean, month_numbers,
                                        init_output['rotation_matrices'], params)

    # Reshape to grid
    result = result.T.reshape((downscaling_factors['lat'],
                                downscaling_factors['lon'],
                                month_numbers['sim_fine'].size))

    return result
