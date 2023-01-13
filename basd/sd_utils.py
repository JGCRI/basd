import numpy as np
import scipy.linalg as spl

import basd.utils as util


def assert_validity_of_months(months):
    """
    Raises an assertion error if any of the numbers in months is not in
    {1,...,12}.

    Parameters
    ----------
    months: np.Array
        Sequence of ints representing calendar months.
    """
    months_allowed = np.arange(1, 13)
    for month in months:
        assert month in months_allowed, f'found {month} in months'


def only_missing_values_in_at_least_one_time_series(data):
    """
    Tests whether there are only missing values in at least one time series
    included in data.

    Parameters
    ----------
    data: dict of array or ndarray
        Keys : 'obs_fine', 'sim_coarse', 'sim_fine'.
        Values : array (for key 'sim_coarse') or ndarray representing climate
        data per coarse grid cell. The first axis is considered the time axis.

    Returns
    -------
    result: bool
        Test result.

    """
    # TODO: This should work regardless of where time axis is
    for key, a in data.items():
        assert a.ndim in [1, 2], f'{key} array has {a.ndim} dimensions'
        if isinstance(a, np.ma.MaskedArray):
            m = a.mask
            if isinstance(m, np.ndarray):
                if a.ndim == 1:
                    if np.all(m):
                        return True
                else:
                    if np.any(np.all(m, axis=0)):
                        return True
            else:
                if m:
                    return True

    return False


def weighted_sum_preserving_mbcn(obs_fine, sim_coarse, sim_fine,
                                 sum_weights, rotation_matrices=None,
                                 n_quantiles=50):
    """
    Applies the core of the modified MBCn algorithm for statistical downscaling
    as described in Lange (2019) <https://doi.org/10.5194/gmd-12-3055-2019>.

    Parameters
    ----------
    obs_fine: (M,N) ndarray
        2D array with M rows, where M is the number of time steps, and N cols,
        being the number of fine cells associated with the given coarse cell.
        This is the observational data
    sim_coarse: array
        1D array of length M, being the number of time-steps
    sim_fine: (M,N) ndarray
        2D array with M rows, where M is the number of time steps, and N cols,
        being the number of fine cells associated with the given coarse cell.
        This is the simulated data, interpolated to the observational spatial resolution
    sum_weights: np.Array
        1D array of length N, grid cell area weights
    rotation_matrices: list , optional
        (N,N) ndarrays in a list. These are orthogonal matrices defining
        a sequence of rotations
    n_quantiles: int, optional
        Number of quantile-quantile pairs used for non-parametric quantile mapping.

    Returns
    -------
    sim_fine: (M,N) ndarray
        2D array with M rows, where M is the number of time steps, and N cols,
        being the number of fine cells associated with the given coarse cell.
        This is the result of the modified MBCn algorithm
    """
    # Set rotation matrix list to empty list if not provided
    # This will make algorithm perform just one iteration
    if rotation_matrices is None:
        rotation_matrices = []

    # initialize total rotation matrix
    n_variables = sum_weights.size
    o_total = np.diag(np.ones(n_variables))

    # p-values in percent for non-parametric quantile mapping
    p = np.linspace(0., 1., n_quantiles + 1)

    # normalise the sum weights vector to length 1
    sum_weights = sum_weights / np.sqrt(np.sum(np.square(sum_weights)))

    # Initial coarse value
    # print(f'Initial Coarse Value: {sim_coarse[0]}')
    # print(f'Initial Agg fine value: {np.dot(sim_fine, sum_weights)[0]}')

    # rescale x_sim_coarse for initial step of algorithm
    # sim_coarse = sim_coarse * np.sum(sum_weights)
    # print(f'Initial Coarse Value Scaled: {sim_coarse[0]}')

    # Need to iterate at least twice (one rotation, and then reversing that rotation)
    # Will iterate additionally however many extra rotation matrices are provided
    n_loops = len(rotation_matrices) + 2
    for i in range(n_loops):
        # On the first iteration
        if not i:
            # rotate to the sum axis
            # Step 3a.1 in <https://doi.org/10.5194/gmd-12-3055-2019>
            o = generate_rotation_matrix_fixed_first_axis(sum_weights)

        # On the last iteration
        elif i == n_loops - 1:
            # Step 3c.1 in <https://doi.org/10.5194/gmd-12-3055-2019>
            # rotate back to original axes for last qm
            o = o_total.T

        # On any other iteration
        else:
            # do random rotation
            o = rotation_matrices[i - 1]

        # compute total rotation
        o_total = np.dot(o_total, o)

        # rotate data
        sim_fine = np.dot(sim_fine, o)
        obs_fine = np.dot(obs_fine, o)
        sum_weights = np.dot(sum_weights, o)

        # On the first iteration
        if not i:
            # restore simulated values at coarse grid scale
            # Step 3a.2 in <https://doi.org/10.5194/gmd-12-3055-2019>
            # TODO: This step is weird because we are no longer using the result of the
            #       rotation. Explicitly changing the result of the rotation. Then, the final
            #       rotation back, shouldn't be exactly the same.
            sim_fine[:, 0] = sim_coarse

            # quantile map observations to values at coarse grid scale
            # Step 3a.3 in <https://doi.org/10.5194/gmd-12-3055-2019>
            q_sim = util.percentile1d(sim_coarse, p)
            q_obs = util.percentile1d(obs_fine[:, 0], p)
            obs_fine[:, 0] = util.map_quantiles_non_parametric_with_constant_extrapolation(
                    obs_fine[:, 0], q_obs, q_sim)

        # For every other iteration
        else:
            # Save output of previous step
            x_sim_previous = sim_fine.copy()

            # do uni-variate non-parametric quantile mapping for every variable
            # Step 3b.2 of <https://doi.org/10.5194/gmd-12-3055-2019>
            for j in range(n_variables):
                q_sim = util.percentile1d(sim_fine[:, j], p)
                q_obs = util.percentile1d(obs_fine[:, j], p)
                sim_fine[:, j] = util.map_quantiles_non_parametric_with_constant_extrapolation(
                    sim_fine[:, j], q_sim, q_obs)

            # preserve weighted sum of original variables
            # Step 3b.3 of <https://doi.org/10.5194/gmd-12-3055-2019>
            # if i < n_loops - 1:
            sim_fine -= np.outer(np.dot(
                sim_fine - x_sim_previous, sum_weights), sum_weights)

        print(f'Agg fine value at iteration {i}: {np.dot(sim_fine, sum_weights)[0]}')

    return sim_fine


def generate_rotation_matrix_fixed_first_axis(v, transpose=False):
    """
        Generates an n x n orthogonal matrix whose first row or column is equal to
        v, a unit vector, and whose other rows or columns are found by Gram-Schmidt
        orthogonalization of sum_weights and the standard unit vectors except the first.

        Parameters
        ----------
        v: (n,) array
            Array of n non-zero numbers.
        transpose: boolean, optional
            If True/False generate an n x n orthogonal matrix whose first row/column
            is equal to v/|v|.

        Returns
        -------
        m: (n,n) ndarray
            Rotation matrix.
        """
    assert np.all(v > 0), 'all elements of v have to be positive'

    # generate matrix of vectors that span the R^n with v being the first vector
    a = np.diag(np.ones_like(v))
    a[:, 0] = v

    # use QR decomposition for Gram-Schmidt orthogonalization of these vectors
    qr = spl.qr(a)
    q = qr[0]

    return -q.T if transpose else -q
