from joblib import Parallel, delayed
import numpy as np
import scipy.linalg as spl
import xarray as xr

from basd.ba_params import Parameters
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

        # Set downscaled grid as copy of fine grid for now
        self.sim_fine = obs_fine

        # Dictionary of datasets to iterate through easily
        self.datasets = {
            'obs_fine': self.obs_fine,
            'sim_coarse': self.sim_coarse,
            'sim_fine': self.sim_fine
        }

        # Set dimension names to lat, lon, time
        self.datasets = util.set_dim_names(self.datasets)
        self.obs_fine = self.datasets['obs_fine']
        self.sim_coarse = self.datasets['sim_coarse']
        self.sim_fine = self.datasets['sim_fine']

        # Analyze input grids
        downscaling_factors, circular, ascending = self.analyze_input_grids()

        # TODO: Perhaps move the sum_weights and CRE_matrix generation to the downscaling function
        #       rather than happening on init.

        # TODO: Actually figure out the remapping/resolution matching process

        # Grid cell weights by global area
        sum_weights = self.grid_cell_weights()

        # get list of rotation matrices to be used for all locations and months
        if params.randomization_seed is not None:
            np.random.seed(params.randomization_seed)
        rotation_matrices = [generate_cre_matrix(int(np.prod(downscaling_factors)))
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
        ascending: dict
        circular: dict
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
        assert np.all(coarse_lat_deltas == coarse_lat_deltas[0]) and np.all(
            fine_lat_deltas == fine_lat_deltas[0]), f'Resolution should be constant for all latitude'
        assert np.all(coarse_lon_deltas == coarse_lon_deltas[0]) and np.all(
            fine_lon_deltas == fine_lon_deltas[0]), f'Resolution should be constant for all longitude'

        # Determine if sequences are circular
        circular = {
            'lat': False,
            'lon': np.allclose(coarse_lons[0] - coarse_lon_deltas[0] + 360 * np.sign(coarse_lon_deltas[0]),
                               coarse_lons[-1])
        }

        # Determine if sequences are increasing
        ascending = {
            'lat': coarse_lat_deltas[0] > 0,
            'lon': coarse_lon_deltas[0] > 0
        }

        # Save the scaling factors
        scale_factors = {
            'lat': f_lat,
            'lon': f_lon
        }

        return scale_factors, circular, ascending

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

    def downscale(self, n_jobs: int = 1):
        # Get days, months and years data
        days, month_numbers, years = util.time_scraping(self.datasets)

        # Iterate through coarse cells in parallel
        i_locations = np.ndindex(self.coarse_sizes['lat'], self.coarse_sizes['lon'])

        # Find and save results into adjusted DataSet
        results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10) \
            (delayed(downscale_one_location_parallel)(
                self.obs_fine[self.variable][dict(lat=i_loc[0], lon=i_loc[1])],
                self.sim_coarse[self.variable][dict(lat=i_loc[0], lon=i_loc[1])],
                self.params,
                days,
                month_numbers,
                years) for i_loc in i_locations)

        i_locations = np.ndindex(self.coarse_sizes['lat'], self.coarse_sizes['lon'])
        for i, i_loc in enumerate(i_locations):
            self.sim_fine[self.variable][dict(lat=i_loc[0], lon=i_loc[1])] = results[i]

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


def downscale_one_location_parallel(obs_fine, sim_coarse):
    """
    Downscales a single coarse grid cell to the resolution of the fine observational grid,
    at the given location.

    Parameters
    ----------
    obs_fine: np.Array
        Observational data at fine resolution contained in the given coarse cell
    sim_coarse: np.Array
        Simulated data at coarse resolution at the given location

    Returns
    -------
    sim_fine: np.Array
        Data for all fine cells in the given coarse cell over the full time period
    """
    return obs_fine + sim_coarse


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
