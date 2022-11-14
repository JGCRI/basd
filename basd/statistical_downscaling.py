import warnings

from joblib import Parallel, delayed
import numpy as np
import scipy.linalg as spl
import xarray as xr

from basd.ba_params import Parameters
import basd.regridding as rg
import basd.sd_utils as sdu
import basd.utils as util


class Downscaler:
    def __init__(self,
                 obs_fine: xr.Dataset,
                 sim_coarse: xr.Dataset,
                 variable: str,
                 params: Parameters):
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
        """

        # Set base input data
        self.obs_fine = obs_fine.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
        self.sim_coarse = sim_coarse.convert_calendar('proleptic_gregorian', align_on='date', missing=np.nan)
        self.params = params
        self.variable = variable

        # Input calendar of the simulation model
        self.input_calendar = sim_coarse.time.dt.calendar

        # Dictionary of datasets to iterate through easily
        self.datasets = {
            'obs_fine': self.obs_fine,
            'sim_coarse': self.sim_coarse,
            'sim_fine': self.obs_fine
        }

        # Set dimension names to lat, lon, time
        self.datasets = util.set_dim_names(self.datasets)
        self.obs_fine = self.datasets['obs_fine']
        self.sim_coarse = self.datasets['sim_coarse']
        self.sim_fine = self.datasets['sim_fine']

        # TODO: Perhaps move the sum_weights and CRE_matrix generation to the downscaling function
        #       rather than happening on init.

        # TODO: Actually figure out the remapping/resolution matching process
        self.sim_coarse = rg.reproject_for_integer_factors(self.obs_fine, self.sim_coarse)

        # Analyze input grids
        self.downscaling_factors = self.analyze_input_grids()

        # Set downscaled grid as copy of fine grid for now
        self.sim_fine = rg.interpolate_for_downscaling(self.obs_fine, self.sim_coarse)

        # Update dictionary
        self.datasets['sim_fine'] = self.sim_fine

        # Grid cell weights by global area
        sum_weights = self.grid_cell_weights()

        # get list of rotation matrices to be used for all locations and months
        if params.randomization_seed is not None:
            np.random.seed(params.randomization_seed)
        rotation_matrices = [generate_cre_matrix(self.downscaling_factors['lat'] * self.downscaling_factors['lon'])
                             for _ in range(params.n_iterations)]

        self.rotation_matrices = rotation_matrices
        self.sum_weights = sum_weights

        # Size of coarse data
        self.coarse_sizes = self.sim_coarse.sizes

    def analyze_input_grids(self):
        """
        Asserts that grids are of compatible sizes, and returns scaling factors,
        direction of coordinate sequence, and whether the sequence is circular

        Returns
        -------
        downscaling_factors: dict
        """
        # Coordinate sequences
        fine_lats = self.obs_fine.coords['lat'].values
        fine_lons = self.obs_fine.coords['lon'].values
        coarse_lats = self.sim_coarse.coords['lat'].values
        coarse_lons = self.sim_coarse.coords['lon'].values

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

    def grid_cell_weights(self):
        """
        Function for finding the weight of each grid cell based on global grid area

        Returns
        -------
        sum_weights: np.array
        """
        lats = self.obs_fine.coords['lat'].values
        weight_by_lat = np.cos(np.deg2rad(lats))

        return weight_by_lat

    def downscale_one_location(self, i_loc: dict):
        """
        Function to downscale a single coarse grid cell

        Parameters
        ----------
        i_loc: dict
            Dictionary of lat, lon index, for coarse cell

        Returns
        -------
        sim_fine_loc: ndarray
            3D array of downscaled coarse grid. Size is latitude scaling factor by longitude scaling
            factor by time
        """

        # Get days, months and years data
        days, month_numbers, years = util.time_scraping(self.datasets)

        # Get data at location
        data, sum_weights_loc = get_data_at_loc(i_loc, self.variable,
                                                self.obs_fine, self.sim_coarse, self.sim_fine,
                                                month_numbers, self.downscaling_factors, self.sum_weights)

        # abort here if there are only missing values in at least one time series
        # do not abort though if the if_all_invalid_use option has been specified
        if np.isnan(self.params.if_all_invalid_use):
            if sdu.only_missing_values_in_at_least_one_time_series(data):
                warnings.warn(f'{i_loc} skipped due to missing data')
                return None

        # compute mean value over all time steps for invalid value sampling
        long_term_mean = {}
        for key, d in data.items():
            long_term_mean[key] = util.average_valid_values(d, self.params.if_all_invalid_use,
                                                            self.params.lower_bound, self.params.lower_threshold,
                                                            self.params.upper_bound, self.params.upper_threshold)

        # Result in flattened format
        result = downscale_month_by_month(data, sum_weights_loc,
                                          long_term_mean, month_numbers,
                                          self.rotation_matrices, self.params)

        # Reshape to grid
        result = result.T.reshape((self.downscaling_factors['lat'],
                                   self.downscaling_factors['lon'],
                                   month_numbers['sim_fine'].size))

        return result

    def downscale(self, n_jobs: int = 1, path: str=None):
        # Get days, months and years data
        days, month_numbers, years = util.time_scraping(self.datasets)

        # Iterate through coarse cells in parallel
        i_locations = np.ndindex(self.coarse_sizes['lat'], self.coarse_sizes['lon'])

        # Find and save results into adjusted DataSet
        results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10) \
            (delayed(downscale_one_location_parallel)(
                dict(lat=i_loc[0], lon=i_loc[1]), self.variable, self.params,
                self.obs_fine, self.sim_coarse, self.sim_fine,
                month_numbers, self.downscaling_factors, self.sum_weights,
                self.rotation_matrices) for i_loc in i_locations)

        i_locations = np.ndindex(self.coarse_sizes['lat'], self.coarse_sizes['lon'])
        for i, i_loc in enumerate(i_locations):
            # Range of indexes of fine cells
            lat_indexes = np.arange(i_loc[0] * self.downscaling_factors['lat'],
                                    (i_loc[0] + 1) * self.downscaling_factors['lat'])
            lon_indexes = np.arange(i_loc[1] * self.downscaling_factors['lon'],
                                    (i_loc[1] + 1) * self.downscaling_factors['lon'])
            # Place results in sim_fine. Note that sim_fine.values are in time, lat, lon order
            # Meanwhile, results are in lat, lon, time order. Thus, we use numpy's transpose to
            # reorder the dimensions
            self.sim_fine[self.variable][dict(lat=lat_indexes, lon=lon_indexes)].values = \
                results[i].transpose((2, 0, 1))

        if path:
            self.save_downscale_nc(path)

        return self.sim_fine

    def save_downscale_nc(self, path):
        """
        Saves adjusted data to NetCDF file at specific path

        Parameters
        ----------
        path: str
            Location and name of output file
        """
        try:
            self.sim_fine.convert_calendar(self.input_calendar, align_on='date').to_netcdf(path)
        except AttributeError:
            try:
                self.sim_fine.to_netcdf(path)
            except AttributeError:
                AttributeError('Unable to write to NetCDF. Possibly incompatible calendars.')


def get_data_at_loc(loc, variable,
                    obs_fine, sim_coarse, sim_fine,
                    month_numbers, downscaling_factors, sum_weights):
    """
    Function for extracting the data from the desired location and in the desired form

    Parameters
    ----------
    loc
    variable
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
    coarse_values = sim_coarse[variable][loc].values
    data['sim_coarse'] = coarse_values

    # Range of indexes of fine cells
    lat_indexes = np.arange(loc['lat'] * downscaling_factors['lat'],
                            (loc['lat'] + 1) * downscaling_factors['lat'])
    lon_indexes = np.arange(loc['lon'] * downscaling_factors['lon'],
                            (loc['lon'] + 1) * downscaling_factors['lon'])

    # Get the cell weights of relevant latitudes
    sum_weights_loc = sum_weights[lat_indexes].repeat(downscaling_factors['lon'])

    # Value of fine observational cells
    obs_fine_values = obs_fine['pr'][dict(lat=lat_indexes, lon=lon_indexes)].values

    # Values of fine simulated cells
    sim_fine_values = sim_fine['pr'][dict(lat=lat_indexes, lon=lon_indexes)].values

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

    return data, sum_weights_loc


def downscale_one_location_parallel(loc, variable, params,
                                    obs_fine, sim_coarse, sim_fine,
                                    month_numbers, downscaling_factors, sum_weights,
                                    rotation_matrices):
    """
    Downscales a single coarse grid cell to the resolution of the fine observational grid,
    at the given location.

    Parameters
    ----------
    rotation_matrices
    params
    sum_weights
    downscaling_factors
    sim_fine
    variable
    loc
    month_numbers
    obs_fine: np.Array
        Observational data at fine resolution contained in the given coarse cell
    sim_coarse: np.Array
        Simulated data at coarse resolution at the given location

    Returns
    -------
    sim_fine_loc: ndarray
        3D array of downscaled coarse grid. Size is latitude scaling factor by longitude scaling
        factor by time
    """

    # Get data at location
    data, sum_weights_loc = get_data_at_loc(loc, variable,
                                            obs_fine, sim_coarse, sim_fine,
                                            month_numbers, downscaling_factors, sum_weights)

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
    result = result.T.reshape((downscaling_factors['lat'],
                               downscaling_factors['lon'],
                               month_numbers['sim_fine'].size))

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
            valid_values = util.sample_invalid_values(d[m], params.randomization_seed, long_term_mean[key])[0]
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
    Applies the mbcn algorithm with weight preservation. Randomizes censored values

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

    sim_fine = sdu.weighted_sum_preserving_mbcn(data_this_month['obs_fine'],
                                                data_this_month['sim_coarse'],
                                                data_this_month['sim_fine'],
                                                sum_weights,
                                                rotation_matrices,
                                                params.n_quantiles)

    sim_fine = util.randomize_censored_values(sim_fine,
                                              lower_bound, lower_threshold,
                                              upper_bound, upper_threshold,
                                              True, True)
    # TODO: assert no infs or nans

    return sim_fine


# noinspection PyTupleAssignmentBalance
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
