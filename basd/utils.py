import numpy as np
import xarray
import warnings
from pandas import Series
import scipy.interpolate as spi
from netCDF4 import num2date


def ma2a(a, raise_error: bool = False):
    """
    Turns masked array into array, replacing missing values and infs by nans.

    Parameters
    ----------
    a : array or masked array
        Array to convert.
    raise_error : boolean, optional
        Whether to raise an error if missing values, infs or nans are found.

    Returns
    -------
    b : array
        Data array.

    """
    b = np.ma.masked_invalid(a, copy=True)
    if np.any(b.mask):
        if raise_error:
            raise ValueError('found missing values, infs or nans in a')
        else:
            return b.filled(np.nan)
    else:
        return b.data


def analyze_input_nc(dataset, variable):
    """
    Returns coordinate variables associated with the given data variable in
    the given netcdf dataset, after making some assertions.

    Parameters
    ----------
    dataset : Dataset
        NetCDF dataset to analyze.
    variable : str
        Data variable to analyze.

    Returns
    -------
    coords : dict of str : array
        Keys : names of dimensions of data variable.
        Values : values of associated coordinate variables.

    """
    # there must be a variable in dataset with name variable
    dataset_variables = dataset.variables.keys()
    msg = f'could not find variable {variable} in nc file'
    assert variable in dataset_variables, msg

    # there must be coordinate variables for all dimensions of the data variable
    coords = {}
    for dim in dataset[variable].dimensions:
        msg = f'could not find variable {dim} in nc file'
        assert dim in dataset_variables, msg(dim)
        dd = dataset[dim]
        msg = f'variable {dim} should have dimensions ({dim},)'
        assert dd.dimensions == (dim,), msg
        coords[dim] = ma2a(dd[:], True)

    # time must be the last dimension
    assert coords and dim == 'time', 'time must be last dimension'

    # the proleptic gregorian calendar must be used
    msg = 'calendar must be proleptic_gregorian'
    assert 'calendar' in dd.ncattrs(), msg
    assert dd.getncattr('calendar') == 'proleptic_gregorian', msg

    # convert time coordinate values to datetime objects
    # TODO: substitute with a xarray function or other?
    coords[dim] = num2date(list(coords[dim]), dd.units, dd.calendar)

    return coords


def average_valid_values(a, if_all_invalid_use=np.nan,
        lower_bound=None, lower_threshold=None,
        upper_bound=None, upper_threshold=None):
    """
    Returns the average over all valid values in a, where missing/inf/nan values
    are considered invalid, unless there are only invalid values in a, in which
    case if_all_invalid_use is returned. Prior to averaging, values beyond
    threshold are set to the respective bound.

    Parameters
    ----------
    a : array or masked array
        If this is an array then infs and nans in a are replaced.
        If this is a masked array then infs, nans, and missing values in a.data
        are replaced using a.mask to indicate missing values.
    if_all_invalid_use : float, optional
        Used as replacement of invalid values if no valid values can be found.
    lower_bound : float, optional
        Lower bound of values in time series.
    lower_threshold : float, optional
        Lower threshold of values in time series.
    upper_bound : float, optional
        Upper bound of values in time series.
    upper_threshold : float, optional
        Upper threshold of values in time series.

    Returns
    -------
    average : float or array of floats
        Result of averaging. The result is scalar if a is one-dimensional.
        Otherwise, the result is an array containing averages for every location.

    """
    # Find invalid values (inf, nan, missing)
    invalid = np.ma.masked_invalid(a, copy=False).mask

    # If all invalid then return nan
    if np.all(invalid):
        return if_all_invalid_use
    else:
        # Get only valid values
        x = a.copy()[np.logical_not(invalid)]

        # Replace values above/below upper/lower thresholds, with the lower/upper bounds
        if lower_bound is not None and lower_threshold is not None:
            x[x <= lower_threshold] = lower_bound
        if upper_bound is not None and upper_threshold is not None:
            x[x >= upper_threshold] = upper_bound

        # Take the mean
        return np.mean(x)


def window_indices_for_running_bias_adjustment(
        days, window_center, window_width, years=None):
    """
    Returns window indices for data selection for bias adjustment in
    running-window mode.

    Parameters
    ----------
    days : array
        Day-of-year time series associated to data array from which data shall
        be selected using the resulting indices.
    window_center : int
        Day of year at the center of each window.
    window_width : int
        Width of each window in days.
    years : array, optional
        Year time series associated to data array from which data shall
        be selected using the resulting indices. If provided, it is ensured
        that windows do not extend into the following or previous year.

    Returns
    -------
    i_window : array
        Window indices.

    """
    # TODO: Better understand how we index data for running window mode
    i_center = np.where(days == 365)[0] + 1 \
               if window_center == 366 else \
               np.where(days == window_center)[0]
    h = window_width // 2
    if years is None:
        i_window = np.concatenate([np.arange(i-h, i+h+1) for i in i_center])
        i_window = np.sort(np.mod(i_window, days.size))
    else:
        years_unique = np.unique(years)
        if years_unique.size == 1:
            # time series only covers one year
            i = i_center[0]
            i_window = np.mod(np.arange(i-h, i+h+1), days.size)
            i_window = i_window[i_window == np.arange(i-h, i+h+1)]
        else:
            # time series covers multiple years
            i_window_list = []
            for j, i in enumerate(i_center):
                i_this_window = np.mod(np.arange(i-h, i+h+1), days.size)
                y_this_window = years[i_this_window]
                i_this_window = i_this_window[y_this_window == years_unique[j]]
                i_window_list.append(i_this_window)
            i_window = np.concatenate(i_window_list)
    return i_window


def percentile1d(a, p):
    """
    Fast version of np.percentile with linear interpolation for 1d arrays
    inspired by
    <https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/>.

    Parameters
    ----------
    a : array
        Input array.
    p : array
        Percentages expressed as real numbers in [0, 1] for which percentiles
        are computed.

    Returns
    -------
    percentiles : array
        Percentiles

    """
    n = a.size - 1
    b = np.sort(a)
    i = n * p
    i_below = np.floor(i).astype(int)
    w_above = i - i_below
    return b[i_below] * (1. - w_above) + b[i_below + (i_below < n)] * w_above


def sample_invalid_values(a, seed=None, if_all_invalid_use=np.nan, warn=False):
    """
    Replaces missing/inf/nan values in a by if_all_invalid_use or by sampling
    from all other values from the same location.

    Parameters
    ----------
    a : array or masked array
        If this is an array then infs and nans in a are replaced.
        If this is a masked array then infs, nans, and missing values in a.data
        are replaced using a.mask to indicate missing values.
    seed : int, optional
        Used to seed the random number generator before replacing invalid
        values.
    if_all_invalid_use : float or array of floats, optional
        Used as replacement of invalid values if no valid values can be found.
    warn : boolean, optional
        Warn user about replacements being made.

    Returns
    -------
    d_replaced : array
        Result of invalid data replacement.
    l_invalid : array
        Boolean array indicating indices of replacement.

    """
    # make sure types and shapes of a and if_all_invalid_use fit
    space_shape = a.shape[1:]
    if len(space_shape):
        msg = 'expected if_all_invalid_use to be an array'
        assert isinstance(if_all_invalid_use, np.ndarray), msg
        msg = 'shapes of a and if_all_invalid_use do not fit'
        assert if_all_invalid_use.shape == space_shape, msg
    else:
        msg = 'expected if_all_invalid_use to be scalar'
        assert np.isscalar(if_all_invalid_use), msg

    # assert that a is a masked array
    if isinstance(a, np.ma.MaskedArray):
        d = a.data
        m = a.mask
        if not isinstance(m, np.ndarray):
            m = np.empty(a.shape, dtype=bool)
            m[:] = a.mask
    else:
        d = a
        m = np.zeros(a.shape, dtype=bool)

    # look for missing values
    l_invalid = m
    n_missing = np.sum(l_invalid)
    if n_missing:
        msg = 'found %i missing value(s)'%n_missing
        if warn: warnings.warn(msg)

    # look for infs
    l_inf = np.isinf(d)
    n_inf = np.sum(l_inf)
    if n_inf:
        msg = 'found %i inf(s)'%n_inf
        if warn: warnings.warn(msg)
        l_invalid = np.logical_or(l_inf, l_invalid)

    # look for nans
    l_nan = np.isnan(d)
    n_nan = np.sum(l_nan)
    if n_nan:
        msg = 'found %i nan(s)'%n_nan
        if warn: warnings.warn(msg)
        l_invalid = np.logical_or(l_nan, l_invalid)

    # return d if all values are valid
    n_invalid = np.sum(l_invalid)
    if not n_invalid:
        return d, None

    # otherwise replace invalid values location by location
    if len(space_shape):
        d_replaced = np.empty_like(d)
        for i in np.ndindex(space_shape):
            j = (slice(None, None),) + i
            d_replaced[j] = sample_invalid_values_core(
                d[j], seed, if_all_invalid_use[i], warn, l_invalid[j])
    else:
        d_replaced = sample_invalid_values_core(
            d, seed, if_all_invalid_use, warn, l_invalid)

    return d_replaced, l_invalid


def sample_invalid_values_core(d, seed, if_all_invalid_use, warn, l_invalid):
    """
    Replaces missing/inf/nan values in d by if_all_invalid_use or by sampling
    from all other values.

    Parameters
    ----------
    d : array
        Containing values to be replaced.
    seed : int
        Used to seed the random number generator before sampling.
    if_all_invalid_use : float
        Used as replacement of invalid values if no valid values can be found.
    warn : boolean
        Warn user about replacements being made.
    l_invalid : array
        Indicating which values in a are invalid and hence to be replaced.

    Returns
    -------
    d_replaced : array
        Result of invalid data replacement.

    """
    # return d if all values in d are valid
    n_invalid = np.sum(l_invalid)
    if not n_invalid:
        return d

    # no sampling possible if there are no valid values in d
    n_valid = d.size - n_invalid
    if not n_valid:
        msg = 'found no valid value(s)'
        if np.isnan(if_all_invalid_use):
            raise ValueError(msg)
        else:
            msg += ': setting them all to %f'%if_all_invalid_use
            if warn: warnings.warn(msg)
            d_replaced = np.empty_like(d)
            d_replaced[:] = if_all_invalid_use
            return d_replaced

    # replace invalid values by sampling from valid values
    # shuffle sampled values to mimic trend in valid values
    msg = 'replacing %i invalid value(s)'%n_invalid + \
    ' by sampling from %i valid value(s)'%n_valid
    if warn: warnings.warn(msg)
    l_valid = np.logical_not(l_invalid)
    d_valid = d[l_valid]
    if seed is not None: np.random.seed(seed)
    p_sampled = np.random.random_sample(n_invalid)
    d_sampled = percentile1d(d_valid, p_sampled)
    d_replaced = d.copy()
    if n_valid == 1:
        d_replaced[l_invalid] = d_sampled
    else:
        i_valid = np.where(l_valid)[0]
        r_valid = np.argsort(np.argsort(d_valid))
        r_valid_interp1d = spi.interp1d(i_valid, r_valid, fill_value='extrapolate')
        i_sampled = np.where(l_invalid)[0]
        r_sampled = np.argsort(np.argsort(r_valid_interp1d(i_sampled)))
        d_replaced[l_invalid] = np.sort(d_sampled)[r_sampled]
    return d_replaced


def randomize_censored_values_core(y, bound, threshold, inverse, power, lower):
    """
    Randomizes values beyond threshold in y or de-randomizes such formerly
    randomized values. Note that y is changed in-place. The randomization
    algorithm is inspired by <https://stackoverflow.com/questions/47429845/
    rank-with-ties-in-python-when-tie-breaker-is-random>

    Parameters
    ----------
    y : array
        Time series to be (de-)randomized.
    bound : float
        Lower or upper bound of values in time series.
    threshold : float
        Lower or upper threshold of values in time series.
    inverse : boolean
        If True, values beyond threshold in y are set to bound.
        If False, values beyond threshold in y are randomized.
    power : float
        Numbers for randomizing values are drawn from a uniform distribution
        and then taken to this power.
    lower : boolean
        If True/False, consider bound and threshold to be lower/upper bound and
        lower/upper threshold, respectively.

    """
    if lower: i = y <= threshold
    else: i = y >= threshold
    if inverse:
        y[i] = bound
    else:
        n = np.sum(i)
        if n:
            p = np.power(np.random.uniform(0, 1, n), power)
            v = bound + p * (threshold - bound)
            s = Series(y[i])
            r = s.sample(frac=1).rank(method='first').reindex_like(s)
            y[i] = np.sort(v)[r.values.astype(int) - 1]


def randomize_censored_values(x,
                              lower_bound=None, lower_threshold=None,
                              upper_bound=None, upper_threshold=None,
                              inplace=False, inverse=False,
                              seed=None, lower_power=1., upper_power=1.):
    """
    Randomizes values beyond threshold in x or de-randomizes such formerly
    randomized values.

    Parameters
    ----------
    x : array
        Time series to be (de-)randomized.
    lower_bound : float, optional
        Lower bound of values in time series.
    lower_threshold : float, optional
        Lower threshold of values in time series.
    upper_bound : float, optional
        Upper bound of values in time series.
    upper_threshold : float, optional
        Upper threshold of values in time series.
    inplace : boolean, optional
        If True, change x in-place. If False, change a copy of x.
    inverse : boolean, optional
        If True, values beyond thresholds in x are set to the respective bound.
        If False, values beyond thresholds in x are randomized, i.e. values that
        exceed upper_threshold are replaced by random numbers from the
        interval [lower_bound, lower_threshold), and values that fall short
        of lower_threshold are replaced by random numbers from the interval
        (upper_threshold, upper_bound]. The ranks of the censored values are
        preserved using a random tie breaker.
    seed : int, optional
        Used to seed the random number generator before replacing values beyond
        threshold.
    lower_power : float, optional
        Numbers for randomizing values that fall short of lower_threshold are
        drawn from a uniform distribution and then taken to this power.
    upper_power : float, optional
        Numbers for randomizing values that exceed upper_threshold are drawn
        from a uniform distribution and then taken to this power.

    Returns
    -------
    x : array
        Randomized or de-randomized time series.

    """
    y = x if inplace else x.copy()
    if seed is not None:
        np.random.seed(seed)

    # randomize lower values
    if lower_bound is not None and lower_threshold is not None:
        randomize_censored_values_core(
            y, lower_bound, lower_threshold, inverse, lower_power, True)

    # randomize upper values
    if upper_bound is not None and upper_threshold is not None:
        randomize_censored_values_core(
            y, upper_bound, upper_threshold, inverse, upper_power, False)

    return y