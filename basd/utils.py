import numpy as np
import warnings

import pandas as pd
from pandas import Series
import scipy.interpolate as spi
import scipy.linalg as spl
import scipy.stats as sps

# Dictionary of possible distribution params implemented thus far
DISTRIBUTION_PARAMS = {
    'normal': sps.norm,
    'weibull': sps.weibull_min,
    'gamma': sps.gamma,
    'beta': sps.beta,
    'rice': sps.rice
}


def ma2a(a, raise_error: bool = False):
    """
    Turns masked array into array, replacing missing values and infs by nans.

    Parameters
    ----------
    a: array or masked array
        Array to convert.
    raise_error : boolean, optional
        Whether to raise an error if missing values, infs or nans are found.

    Returns
    -------
    b: array
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


def average_valid_values(a, if_all_invalid_use=np.nan,
                         lower_bound=None, lower_threshold=None,
                         upper_bound=None, upper_threshold=None):
    """
    Returns the average over all valid values in 'a', where missing/inf/nan values
    are considered invalid, unless there are only invalid values in 'a', in which
    case if_all_invalid_use is returned. Prior to averaging, values beyond
    threshold are set to the respective bound.

    Parameters
    ----------
    a: array or masked array
        If this is an array then infs and nans in 'a' are replaced.
        If this is a masked array then infs, nans, and missing values in a.data
        are replaced using a.mask to indicate missing values.
    if_all_invalid_use : float, optional
        Used as replacement of invalid values if no valid values can be found.
    lower_bound: float, optional
        Lower bound of values in time series.
    lower_threshold : float, optional
        Lower threshold of values in time series.
    upper_bound: float, optional
        Upper bound of values in time series.
    upper_threshold: float, optional
        Upper threshold of values in time series.

    Returns
    -------
    average: float or array of floats
        Result of averaging. The result is scalar if 'a' is one-dimensional.
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


def window_centers_for_running_bias_adjustment(days, step_size):
    """
    Gives the days to center the windows around in running window mode depending on the step size

    Parameters
    ----------
    days: dict
        Dictionary of arrays that give day of year for each observation in each dataset
    step_size: int
        Number of days used for bias adjustment in running window mode

    Returns
    -------
    days_center: np.Array
        Array of days to center windows around
    """
    # TODO: Understand this better
    days_max = np.max(days['sim_fut'])
    days_mod = days_max % step_size
    days_center_first = 1 + step_size // 2

    # make sure first and last window have the sane length (+/-1)
    if days_mod:
        days_center_first -= (step_size - days_mod) // 2
    days_center = np.arange(days_center_first, days_max + 1, step_size)

    return days_center


def window_indices_for_running_bias_adjustment(
        days, window_center, window_width, years=None):
    """
    Returns window indices for data selection for bias adjustment in
    running-window mode.

    Parameters
    ----------
    days: np.array
        Day-of-year time series associated to data array from which data shall
        be selected using the resulting indices.
    window_center: int
        Day of year at the center of each window.
    window_width: int
        Width of each window in days.
    years: array, optional
        Year time series associated to data array from which data shall
        be selected using the resulting indices. If provided, it is ensured
        that windows do not extend into the following or previous year.

    Returns
    -------
    i_window: np.Array
        Window indices.

    """
    # TODO: Better understand how we index data for running window mode
    i_center = np.where(days == 365)[0] + 1 if window_center == 366 else np.where(days == window_center)[0]
    h = window_width // 2
    if years is None:
        i_window = np.concatenate([np.arange(i - h, i + h + 1) for i in i_center])
        i_window = np.sort(np.mod(i_window, days.size))
    else:
        years_unique = np.unique(years)
        if years_unique.size == 1:
            # time series only covers one year
            i = i_center[0]
            i_window = np.mod(np.arange(i - h, i + h + 1), days.size)
            i_window = i_window[i_window == np.arange(i - h, i + h + 1)]
        else:
            # time series covers multiple years
            i_window_list = []
            for j, i in enumerate(i_center):
                i_this_window = np.mod(np.arange(i - h, i + h + 1), days.size)
                y_this_window = years[i_this_window]
                i_this_window = i_this_window[y_this_window == years_unique[j]]
                i_window_list.append(i_this_window)
            i_window = np.concatenate(i_window_list)
    return i_window


def get_data_in_window(window_center, data_loc, days, years, long_term_mean):
    years_this_window = {}
    data_this_window = {}
    for key, data_arr in data_loc.items():
        # Get indices for data needed for this window
        m = window_indices_for_running_bias_adjustment(days[key], window_center, 31)
        # Associated year for each data point in the resulting data
        years_this_window[key] = years[key][m]
        # Sample invalid values
        replaced, invalid = sample_invalid_values(data_arr.values[m], 1, long_term_mean[key])
        # The actual needed data in this window
        data_this_window[key] = replaced

    return data_this_window, years_this_window


def percentile1d(a, p):
    """
    Fast version of np.percentile with linear interpolation for 1d arrays
    inspired by
    <https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/>.

    Parameters
    ----------
    a: np.array
        Input array.
    p: np.Array
        Percentages expressed as real numbers in [0, 1] for which percentiles
        are computed.

    Returns
    -------
    percentiles: np.array
        Percentiles

    """
    n = a.size - 1
    b = np.sort(a)
    i = n * p
    i_below = np.floor(i).astype(int)
    w_above = i - i_below
    return b[i_below] * (1. - w_above) + b[i_below + (i_below < n)] * w_above


def time_scraping(adjustment):
    """
    Function that turns the time variables for each input dataset into arrays
    of days of the year, month number, and year.

    Parameters
    __________
    adjustment: Adjustment
        Bias adjustment object that holds the input datasets

    Returns
    -------
    days: dict
        Dictionary of numpy arrays that give the day of year for each obs in each dataset
    month_numbers: dict
        Dictionary of numpy arrays that give the month number for each obs in each dataset
    years: dict
        Dictionary of numpy arrays that give the year for each obs in each dataset
    """
    # Scraping the time from the data and turning into pandas date time array
    dates_obs_hist = pd.DatetimeIndex(adjustment.obs_hist['time'].values)
    dates_sim_hist = pd.DatetimeIndex(adjustment.sim_hist['time'].values)
    dates_sim_fut = pd.DatetimeIndex(adjustment.sim_fut['time'].values)

    # Getting the month for each observation
    month_numbers = {
        'obs_hist': dates_obs_hist.month,
        'sim_hist': dates_sim_hist.month,
        'sim_fut': dates_sim_fut.month
    }

    # Getting the year for each observation
    years = {
        'obs_hist': dates_obs_hist.year,
        'sim_hist': dates_sim_hist.year,
        'sim_fut': dates_sim_fut.year
    }

    # Getting the day of the year for each observation
    days = {
        'obs_hist': dates_obs_hist.day_of_year,
        'sim_hist': dates_sim_hist.day_of_year,
        'sim_fut': dates_sim_fut.day_of_year
    }

    return days, month_numbers, years


def sample_invalid_values(a, seed=None, if_all_invalid_use=np.nan, warn=False):
    """
    Replaces missing/inf/nan values in a by if_all_invalid_use or by sampling
    from all other values from the same location.

    Parameters
    ----------
    a: array or masked array
        If this is an array then infs and nans in 'a' are replaced.
        If this is a masked array then infs, nans, and missing values in a.data
        are replaced using a.mask to indicate missing values.
    seed: int, optional
        Used to seed the random number generator before replacing invalid
        values.
    if_all_invalid_use: float or array of floats, optional
        Used as replacement of invalid values if no valid values can be found.
    warn: boolean, optional
        Warn user about replacements being made.

    Returns
    -------
    d_replaced: array
        Result of invalid data replacement.
    l_invalid: array
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

    # assert that 'a' is a masked array
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
        msg = 'found %i missing value(s)' % n_missing
        if warn:
            warnings.warn(msg)

    # look for infs
    l_inf = np.isinf(d)
    n_inf = np.sum(l_inf)
    if n_inf:
        msg = 'found %i inf(s)' % n_inf
        if warn:
            warnings.warn(msg)
        l_invalid = np.logical_or(l_inf, l_invalid)

    # look for nans
    l_nan = np.isnan(d)
    n_nan = np.sum(l_nan)
    if n_nan:
        msg = 'found %i nan(s)' % n_nan
        if warn:
            warnings.warn(msg)
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
    d: np.array
        Containing values to be replaced.
    seed: int
        Used to seed the random number generator before sampling.
    if_all_invalid_use : float
        Used as replacement of invalid values if no valid values can be found.
    warn: boolean
        Warn user about replacements being made.
    l_invalid: np.Array
        Indicating which values in 'a' are invalid and hence to be replaced.

    Returns
    -------
    d_replaced: np.array
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
            msg += ': setting them all to %f' % if_all_invalid_use
            if warn:
                warnings.warn(msg)
            d_replaced = np.empty_like(d)
            d_replaced[:] = if_all_invalid_use
            return d_replaced

    # replace invalid values by sampling from valid values
    # shuffle sampled values to mimic trend in valid values
    msg = 'replacing %i invalid value(s)' % n_invalid + ' by sampling from %i valid value(s)' % n_valid
    if warn:
        warnings.warn(msg)
    l_valid = np.logical_not(l_invalid)
    d_valid = d[l_valid]
    if seed is not None:
        np.random.seed(seed)
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
    y: np.array
        Time series to be (de-)randomized.
    bound: float
        Lower or upper bound of values in time series.
    threshold: float
        Lower or upper threshold of values in time series.
    inverse: boolean
        If True, values beyond threshold in y are set to bound.
        If False, values beyond threshold in y are randomized.
    power: float
        Numbers for randomizing values are drawn from a uniform distribution
        and then taken to this power.
    lower: boolean
        If True/False, consider bound and threshold to be lower/upper bound and
        lower/upper threshold, respectively.

    """
    if lower:
        i = y <= threshold
    else:
        i = y >= threshold
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
    x: np.array
        Time series to be (de-)randomized.
    lower_bound: float, optional
        Lower bound of values in time series.
    lower_threshold: float, optional
        Lower threshold of values in time series.
    upper_bound: float, optional
        Upper bound of values in time series.
    upper_threshold: float, optional
        Upper threshold of values in time series.
    inplace: boolean, optional
        If True, change x in-place. If False, change a copy of x.
    inverse: boolean, optional
        If True, values beyond thresholds in x are set to the respective bound.
        If False, values beyond thresholds in x are randomized, i.e. values that
        exceed upper_threshold are replaced by random numbers from the
        interval [lower_bound, lower_threshold), and values that fall short
        of lower_threshold are replaced by random numbers from the interval
        (upper_threshold, upper_bound]. The ranks of the censored values are
        preserved using a random tiebreaker.
    seed: int, optional
        Used to seed the random number generator before replacing values beyond
        threshold.
    lower_power: float, optional
        Numbers for randomizing values that fall short of lower_threshold are
        drawn from a uniform distribution and then taken to this power.
    upper_power: float, optional
        Numbers for randomizing values that exceed upper_threshold are drawn
        from a uniform distribution and then taken to this power.

    Returns
    -------
    x: np.array
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


def adjust_bias_one_month(data, years, long_term_mean, params):
    # Detrend and randomize censored values
    for key, y in years.items():
        # TODO: Implement detrending

        # Randomizing censored values (above/below thresholds)
        randomize_censored_values(data[key],
                                  params.lower_bound, params.lower_threshold,
                                  params.upper_bound, params.upper_threshold,
                                  True, False, 1)

    # TODO: Implement copula mbcn adjustment?

    y = map_quantiles_parametric_trend_preserving(data, params)

    # TODO: Implement re-introducing the trend

    # TODO: assert to infs or nans

    return y


def get_data_within_thresholds(data, params):
    """
    Returns the time series data that falls within the thresholds. Also, the indexes
    of these observations for the observational data which is needed.

    Parameters
    ----------
    data: dict
        Dictionary of time series arrays for each input dataset
    params: Parameters
        Bias adjustment parameters object

    Returns
    -------
    data_within_threshold: dict
        Dictionary of time series arrays for each input dataset, which only includes values
        within the thresholds
    i_obs_hist: np.Array
        indexes which inform where the observational data is within the thresholds.
    """
    lower = params.lower_bound is not None and params.lower_threshold is not None
    upper = params.upper_bound is not None and params.upper_threshold is not None

    # Full data
    x_obs_hist = data['obs_hist']
    x_sim_hist = data['sim_hist']
    x_sim_fut = data['sim_fut']

    # Indexes assuming all observations within thresholds
    i_obs_hist = np.ones(x_obs_hist.shape, dtype=bool)
    i_sim_hist = np.ones(x_sim_hist.shape, dtype=bool)
    i_sim_fut = np.ones(x_sim_fut.shape, dtype=bool)

    # Indexes after asserting lower threshold
    if lower:
        i_obs_hist = np.logical_and(i_obs_hist, x_obs_hist > params.lower_threshold)
        i_sim_hist = np.logical_and(i_sim_hist, x_sim_hist > params.lower_threshold)
        i_sim_fut = np.logical_and(i_sim_fut, x_sim_fut > params.lower_threshold)
    # Indexes after asserting upper threshold
    if upper:
        i_obs_hist = np.logical_and(i_obs_hist, x_obs_hist < params.upper_threshold)
        i_sim_hist = np.logical_and(i_sim_hist, x_sim_hist < params.upper_threshold)
        i_sim_fut = np.logical_and(i_sim_fut, x_sim_fut < params.upper_threshold)

    data_within_thresholds = {
        'obs_hist': x_obs_hist[i_obs_hist],
        'sim_hist': x_sim_hist[i_sim_hist],
        'sim_fut': x_sim_fut[i_sim_fut]
    }

    return data_within_thresholds, i_obs_hist, i_sim_hist, i_sim_fut


def ccs_transfer_sim2obs(
        x_obs_hist, x_sim_hist, x_sim_fut,
        lower_bound=0., upper_bound=1.):
    """
    Generates pseudo future observation(s) by transfering a simulated climate
    change signal to historical observation(s) respecting the given bounds.

    Parameters
    ----------
    x_obs_hist: float or np.Array
        Historical observation(s).
    x_sim_hist: float or np.Array
        Historical simulation(s).
    x_sim_fut: float or np.Array
        Future simulation(s).
    lower_bound: float, optional
        Lower bound of values in input and output data.
    upper_bound: float, optional
        Upper bound of values in input and output data.

    Returns
    -------
    x_obs_fut: float or array
        Pseudo future observation(s).

    """
    # change scalar inputs to arrays
    if np.isscalar(x_obs_hist):
        x_obs_hist = np.array([x_obs_hist])
    if np.isscalar(x_sim_hist):
        x_sim_hist = np.array([x_sim_hist])
    if np.isscalar(x_sim_fut):
        x_sim_fut = np.array([x_sim_fut])

    # check input
    assert lower_bound < upper_bound, 'lower_bound >= upper_bound'
    for x_name, x in zip(['x_obs_hist', 'x_sim_hist', 'x_sim_fut'],
                         [x_obs_hist, x_sim_hist, x_sim_fut]):
        assert np.all(x >= lower_bound), 'found ' + x_name + ' < lower_bound'
        assert np.all(x <= upper_bound), 'found ' + x_name + ' > upper_bound'

    # compute x_obs_fut
    i_neg_bias = x_sim_hist < x_obs_hist
    i_zero_bias = x_sim_hist == x_obs_hist
    i_pos_bias = x_sim_hist > x_obs_hist
    i_additive = np.logical_or(
        np.logical_and(i_neg_bias, x_sim_fut < x_sim_hist),
        np.logical_and(i_pos_bias, x_sim_fut > x_sim_hist))
    x_obs_fut = np.empty_like(x_obs_hist)
    x_obs_fut[i_neg_bias] = upper_bound - \
                            (upper_bound - x_obs_hist[i_neg_bias]) * \
                            (upper_bound - x_sim_fut[i_neg_bias]) / \
                            (upper_bound - x_sim_hist[i_neg_bias])
    x_obs_fut[i_zero_bias] = x_sim_fut[i_zero_bias]
    x_obs_fut[i_pos_bias] = lower_bound + \
                            (x_obs_hist[i_pos_bias] - lower_bound) * \
                            (x_sim_fut[i_pos_bias] - lower_bound) / \
                            (x_sim_hist[i_pos_bias] - lower_bound)
    x_obs_fut[i_additive] = x_obs_hist[i_additive] + \
                            x_sim_fut[i_additive] - x_sim_hist[i_additive]

    # make sure x_obs_fut is within bounds
    x_obs_fut = np.maximum(lower_bound, np.minimum(upper_bound, x_obs_fut))

    return x_obs_fut[0] if x_obs_fut.size == 1 else x_obs_fut


def extreme_value_probabilities(data, params, lower, upper):
    """
    Returns the extreme value probabilities if lower or upper thresholds for the input time
    series

    Parameters
    ----------
    data: dict
    params: Parameters
    lower: bool
    upper: bool

    Returns
    -------
    p_lower_target: float
    p_upper_target: float
    p_lower_or_upper_target: float
    """
    # Data from dict
    x_obs_hist = data['obs_hist']
    x_sim_hist = data['sim_hist']
    x_sim_fut = data['sim_fut']

    # Set to None in case non-bounded
    p_lower_target = None
    p_upper_target = None
    p_lower_or_upper_target = None

    # If lower bounded
    if lower:
        def p_lower(x):
            return np.mean(x <= params.lower_threshold)

        if params.trendless_bound_frequency:
            p_lower_target = p_lower(x_obs_hist)
        else:
            p_lower_target = ccs_transfer_sim2obs(p_lower(x_obs_hist), p_lower(x_sim_hist), p_lower(x_sim_fut))
    # If upper bounded
    if upper:
        def p_upper(x):
            return np.mean(x >= params.upper_threshold)

        if params.trendless_bound_frequency:
            p_upper_target = p_upper(x_obs_hist)
        else:
            p_upper_target = ccs_transfer_sim2obs(p_upper(x_obs_hist), p_upper(x_sim_hist), p_upper(x_sim_fut))
    # If lower and upper bounded
    if lower and upper:
        p_lower_or_upper_target = p_lower_target + p_upper_target
        if p_lower_or_upper_target > 1 + 1e-10:
            msg = 'sum of p_lower_target and p_upper_target exceeds one'
            warnings.warn(msg)
            p_lower_target /= p_lower_or_upper_target
            p_upper_target /= p_lower_or_upper_target

    return p_lower_target, p_upper_target, p_lower_or_upper_target


def indexes_to_map(x_source, x_target, y, params, p_lower_target, p_upper_target, lower, upper):
    """
    Finds the indexes of values that need to be mapped, and sets y to lower or
    upper bound when thresholds are exceeded

    Parameters
    ----------
    x_source: np.Array
    x_target: np.Array
    y: np.Array
    params: Parameters
    p_lower_target: float
    p_upper_target: float
    lower: bool
    upper: bool

    Returns
    -------
    y: np.Array
    i_source: np.Array
    i_target: np.Array
    """
    i_source = np.ones(x_source.shape, dtype=bool)
    i_target = np.ones(x_target.shape, dtype=bool)
    if lower:
        # make sure that lower_threshold_source < x_source
        # because otherwise sps.beta.ppf does not work
        lower_threshold_source = \
            percentile1d(x_source, np.array([p_lower_target]))[0] \
                if p_lower_target > 0 else params.lower_bound if not upper else \
                params.lower_bound - 1e-10 * (params.upper_bound - params.lower_bound)
        i_lower = x_source <= lower_threshold_source
        i_source = np.logical_and(i_source, np.logical_not(i_lower))
        i_target = np.logical_and(i_target, x_target > params.lower_threshold)
        y[i_lower] = params.lower_bound
    if upper:
        # make sure that x_source < upper_threshold_source
        # because otherwise sps.beta.ppf does not work
        upper_threshold_source = \
            percentile1d(x_source, np.array([1. - p_upper_target]))[0] \
                if p_upper_target > 0 else params.upper_bound if not lower else \
                params.upper_bound + 1e-10 * (params.upper_bound - params.lower_bound)
        i_upper = x_source >= upper_threshold_source
        i_source = np.logical_and(i_source, np.logical_not(i_upper))
        i_target = np.logical_and(i_target, x_target < params.upper_threshold)
        y[i_upper] = params.upper_bound

    return y, i_source, i_target


def map_quantiles_non_parametric_brute_force(x, y):
    """
    Quantile-map x to y using the empirical CDFs of x and y.

    Parameters
    ----------
    x: array
        Simulated time series.
    y: array
        Observed time series.

    Returns
    -------
    z: array
        Result of quantile mapping.

    """
    if x.size == 0:
        msg = 'found no values in x: returning x'
        warnings.warn(msg)
        return x

    if np.unique(y).size < 2:
        msg = 'found fewer then 2 different values in y: returning x'
        warnings.warn(msg)
        return x

    p_x = (sps.rankdata(x) - 1.) / x.size  # percent points of x
    p_y = np.linspace(0., 1., y.size)  # percent points of sorted y
    z = np.interp(p_x, p_y, np.sort(y))  # quantile mapping
    return z


def map_quantiles_non_parametric_with_constant_extrapolation():
    return True


def map_quantiles_core(x_source, x_target, y, i_source, i_target, i_sim_fut, params):
    # Determine if distribution has bounds
    lower = params.lower_bound is not None and params.lower_threshold is not None
    upper = params.upper_bound is not None and params.upper_threshold is not None

    # Get distribution parameters if one of the implemented distributions
    # Otherwise we will use non-parametric quantile mapping
    spsdotwhat = None
    if params.distribution in DISTRIBUTION_PARAMS.keys():
        spsdotwhat = DISTRIBUTION_PARAMS[params.distribution]

    # use the within-threshold values of x_sim_fut for the source
    # distribution fitting
    x_source_fit = x_source[i_sim_fut]
    x_target_fit = x_target[i_target]

    if spsdotwhat is None:
        # prepare non-parametric quantile mapping
        x_source_map = x_source[i_source]
        shape_loc_scale_source = None
        shape_loc_scale_target = None
    else:
        # prepare parametric quantile mapping
        if lower or upper:
            # map the values in x_source to be quantile-mapped such that
            # their empirical distribution matches the empirical
            # distribution of the within-threshold values of x_sim_fut
            x_source_map = map_quantiles_non_parametric_brute_force(x_source[i_source], x_source_fit)
        else:
            x_source_map = x_source

        # fit distributions to x_source and x_target
        # TODO: Create function which tries to do this (MLE fit) then method of moments
        #   and then non-parametric if both of those fits fail
        shape_loc_scale_source = spsdotwhat.fit(x_source_fit)
        shape_loc_scale_target = spsdotwhat.fit(x_target_fit)

    # do non-parametric quantile mapping if needed
    if shape_loc_scale_source is None or shape_loc_scale_target is None:
        # This is the case when parametric was desired but failed
        if spsdotwhat is not None:
            msg = 'Parametric quantile mapping failed. Performing non-parametric quantile mapping instead.'
            warnings.warn(msg)
        p_zeroone = np.linspace(0., 1., params.n_quantiles + 1)
        q_source_fit = percentile1d(x_source_map, p_zeroone)
        q_target_fit = percentile1d(x_target_fit, p_zeroone)
        y[i_source] = map_quantiles_non_parametric_with_constant_extrapolation(x_source_map, q_source_fit, q_target_fit)

        # If doing non-parametric mapping, return here
        return y

    # From here, assuming that parametric fit was successful
    # Here are the p-values of the simulated future values according to the
    # fit for the simulated future values
    p_source = spsdotwhat.cdf(x_source_map, *shape_loc_scale_source)
    # Limiting the p-values to not be too small or close to 1
    p_source = np.maximum(params.p_value_eps, np.minimum(1-params.p_value_eps, p_source))

    # Compute target p-values
    if params.adjust_p_values:
        # TODO: Implement this case
        raise Exception('Adjusting p-values not yet implemented')
    else:
        p_target = p_source

    # ppf is inverse CDF function. Here we take the p-values of the target and translate
    # them to data according to the target distribution
    y[i_source] = spsdotwhat.ppf(p_target, *shape_loc_scale_target)

    return y


def map_quantiles_parametric_trend_preserving(data, params):
    """
    Adjusts biases using the trend-preserving parametric quantile mapping method
    described in Lange (2019) <https://doi.org/10.5194/gmd-12-3055-2019>.

    Parameters
    ----------
    data: dict
        Dictionary of arrays for each input data representing time series of climate data
    params: Parameters
        Bias adjustment parameters object

    Returns
    -------
    x_sim_fut_ba: np.Array
        Result of bias adjustment.

    """
    lower = params.lower_bound is not None and params.lower_threshold is not None
    upper = params.upper_bound is not None and params.upper_threshold is not None

    # Get time series arrays from dict
    x_obs_hist = data['obs_hist']
    x_sim_hist = data['sim_hist']
    x_sim_fut = data['sim_fut']

    # Get data within thresholds and indexes of these observations
    data_within_thresholds, i_obs_hist, i_sim_hist, i_sim_fut = get_data_within_thresholds(data, params)

    # Non parametric quantile mapping
    # Use all values if unconditional_css_transfer
    if params.unconditional_ccs_transfer:
        x_target = map_quantiles_non_parametric_trend_preserving(data, params)
    else:
        # use only values within thresholds
        x_target = x_obs_hist.copy()
        x_target[i_obs_hist] = map_quantiles_non_parametric_trend_preserving(data_within_thresholds, params)

    # determine extreme value probabilities of future obs
    p_lower_target, p_upper_target, p_lower_or_upper_target = extreme_value_probabilities(data, params, lower, upper)

    # do a parametric quantile mapping of the values within thresholds
    x_source = x_sim_fut
    y = x_source.copy()

    # determine indices of values to be mapped
    y, i_source, i_target = indexes_to_map(x_source, x_target, y, params,
                                           p_lower_target, p_upper_target,
                                           lower, upper)

    # break here if target distributions cannot be determined
    if not np.any(i_target):
        msg = 'unable to do any quantile mapping' \
              + ': leaving %i value(s) unadjusted' % np.sum(i_source)
        warnings.warn(msg)
        return y

    # map quantiles
    result = map_quantiles_core(x_source, x_target, y, i_source, i_target, i_sim_fut, params)

    return result


def map_quantiles_non_parametric_trend_preserving(data, params):
    """
    Adjusts biases with a modified version of the quantile delta mapping by
    Cannon (2015) <https://doi.org/10.1175/JCLI-D-14-00754.1> or uses this
    method to transfer a simulated climate change signal to observations.

    Parameters
    ----------
    data: Adjustment
    params: Parameters

    Returns
    -------
    y: array
        Result of quantile mapping or climate change signal transfer.

    """
    # Get time series arrays from dict
    x_obs_hist = data['obs_hist']
    x_sim_hist = data['sim_hist']
    x_sim_fut = data['sim_fut']

    # make sure there are enough input data for quantile delta mapping
    # reduce n_quantiles if necessary
    assert params.n_quantiles > 0, 'n_quantiles <= 0'
    n = min([params.n_quantiles + 1, x_obs_hist.size, x_sim_hist.size, x_sim_fut.size])
    if n < 2:
        if params.adjust_obs:
            msg = 'not enough input data: returning x_obs_hist'
            y = x_obs_hist
        else:
            msg = 'not enough input data: returning x_sim_fut'
            y = x_sim_fut
        warnings.warn(msg)
        return y
    elif n < params.n_quantiles + 1:
        msg = 'due to little input data: reducing n_quantiles to %i' % (n - 1)
        warnings.warn(msg)
    p_zeroone = np.linspace(0., 1., n)

    # compute quantiles of input data
    q_obs_hist = percentile1d(x_obs_hist, p_zeroone)
    q_sim_hist = percentile1d(x_sim_hist, p_zeroone)
    q_sim_fut = percentile1d(x_sim_fut, p_zeroone)

    # compute quantiles needed for quantile delta mapping
    if params.adjust_obs:
        p = np.interp(x_obs_hist, q_obs_hist, p_zeroone)
    else:
        p = np.interp(x_sim_fut, q_sim_fut, p_zeroone)
    F_sim_fut_inv = np.interp(p, p_zeroone, q_sim_fut)
    F_sim_hist_inv = np.interp(p, p_zeroone, q_sim_hist)
    F_obs_hist_inv = np.interp(p, p_zeroone, q_obs_hist)

    # do augmented quantile delta mapping
    if params.trend_preservation == 'bounded':
        msg = 'lower_bound or upper_bound not specified'
        assert params.lower_bound is not None and params.upper_bound is not None, msg
        assert params.lower_bound < params.upper_bound, 'lower_bound >= upper_bound'
        y = ccs_transfer_sim2obs(
            F_obs_hist_inv, F_sim_hist_inv, F_sim_fut_inv,
            params.lower_bound, params.upper_bound)
    elif params.trend_preservation in ['mixed', 'multiplicative']:
        assert params.max_change_factor > 1, 'max_change_factor <= 1'
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(F_sim_hist_inv == 0, 1., F_sim_fut_inv / F_sim_hist_inv)
            y[y > params.max_change_factor] = params.max_change_factor
            y[y < 1. / params.max_change_factor] = 1. / params.max_change_factor
        y *= F_obs_hist_inv
        if params.trend_preservation == 'mixed':  # if not then we are done here
            assert params.max_adjustment_factor > 1, 'max_adjustment_factor <= 1'
            y_additive = F_obs_hist_inv + F_sim_fut_inv - F_sim_hist_inv
            fraction_multiplicative = np.zeros_like(y)
            fraction_multiplicative[F_sim_hist_inv >= F_obs_hist_inv] = 1.
            i_transition = np.logical_and(F_sim_hist_inv < F_obs_hist_inv,
                                          F_obs_hist_inv < params.max_adjustment_factor * F_sim_hist_inv)
            fraction_multiplicative[i_transition] = .5 * (1. +
                                                          np.cos((F_obs_hist_inv[i_transition] /
                                                                  F_sim_hist_inv[i_transition] - 1.) *
                                                                 np.pi / (params.max_adjustment_factor - 1.)))
            y = fraction_multiplicative * y + (1. -
                                               fraction_multiplicative) * y_additive
    elif params.trend_preservation == 'additive':
        y = F_obs_hist_inv + F_sim_fut_inv - F_sim_hist_inv
    else:
        msg = 'trend_preservation = ' + params.trend_preservation + ' not supported'
        raise AssertionError(msg)

    return y
