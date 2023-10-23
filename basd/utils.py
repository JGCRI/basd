import numpy as np
import warnings
import pandas as pd
from pandas import Series
import scipy.interpolate as spi
import scipy.stats as sps
from scipy.signal import convolve

from basd.ba_params import Parameters

# Dictionary of possible distribution params implemented thus far
DISTRIBUTION_PARAMS = {
    'normal': sps.norm,
    'weibull': sps.weibull_min,
    'gamma': sps.gamma,
    'beta': sps.beta,
    'rice': sps.rice
}


def set_dim_names(datasets: dict):
    """
    Makes sure latitude, longitude and time dimensions are present. These are set to be named
    lat, lon, time if not already. Will assume a matching dimension if lat, lon, or time is in
    the respective dimension name.

    Parameters
    ----------
    datasets: dict
        Dictionary of datasets which will have their variables renamed if necessary
    """
    # For each of the datasets rename the dimensions
    for data_name, data in datasets.items():
        for key in data.dims:
            if 'lat' in key.lower():
                datasets[data_name] = data.swap_dims({key: 'lat'})
            elif 'lon' in key.lower():
                datasets[data_name] = data.swap_dims({key: 'lon'})
            elif 'time' in key.lower():
                datasets[data_name] = data.swap_dims({key: 'time'})

    # Make sure each required dimension is in each dataset
    for data_name, data in datasets.items():
        msg = f'{data_name} needs a latitude, longitude and time dimension'
        assert all(i in data.dims for i in ['lat', 'lon', 'time']), msg

    return datasets


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


def aggregate_periodic(arr, halfwin, aggregator='mean'):
    """
    Aggregates arr using the given aggregator and a running window of length
    2 * halfwin + 1 assuming that arr is periodic.

    Parameters
    ----------
    arr : np.Array
        Array to be aggregated.
    halfwin : int
        Determines length of running window used for aggregation.
    aggregator : str, optional
        Determines how arr is aggregated along axis 0 for every running window.

    Returns
    -------
    rm : np.ndarray
        Result of aggregation. Same shape as arr.

    """
    # Window should be positive length
    assert halfwin >= 0, 'halfwin < 0'
    if not halfwin:
        return arr

    # Extend a periodically
    # Size of the array
    n = arr.size
    # Making sure window contained within array
    assert n >= halfwin, 'length of a along axis 0 less than halfwin'
    # Wrapping array
    b = np.concatenate((arr[-halfwin:], arr, arr[:halfwin]))

    # Full window width
    window = 2 * halfwin + 1

    # Aggregate using algorithm for max inspired by
    # <http://p-nand-q.com/python/algorithms/searching/max-sliding-window.html>
    if aggregator == 'max':
        c = list(np.maximum.accumulate(b[:window][::-1]))
        rm = np.empty_like(arr)
        rm[0] = c[-1]
        for i in range(n - 1):
            c_new = b[i + window]
            del c[-1]
            for j in range(window - 1):
                if c_new > c[j]:
                    c[j] = c_new
                else:
                    break
            c.insert(0, c_new)
            rm[i + 1] = c[-1]
    elif aggregator == 'mean':
        rm = convolve(b, np.repeat(1. / window, window), 'valid')
    else:
        raise ValueError(f'aggregator {aggregator} not supported')

    return rm


def get_upper_bound_climatology(data_arr, days, halfwin):
    """
    Estimates an annual cycle of upper bounds as running mean values of running
    maximum values of multi-year daily maximum values.

    Parameters
    ----------
    data_arr: np.Array
        Time series for which annual cycles of upper bounds shall be estimated.
    days: np.Array
        Day of the year time series corresponding to d.
    halfwin: int
        Determines length of running windows used for estimation.

    Returns
    -------
    ubc: np.Array
        Upper bound climatology.
    days_unique: np.Array
        Days of the year of upper bound climatology.
    """
    # Each data obs must have associated day of the year, thus shapes must be the same
    assert data_arr.shape == days.shape, 'data and days differ in shape'

    # Get the unique days of the year in order
    unique_days = np.sort(np.unique(days))

    # Warn user if number of unique days is less than 366
    n = unique_days.size
    if n != 366:
        msg = (f'Upper bound climatology only defined for {n} days of the year:'
               ' this may imply an invalid computation of the climatology')
        warnings.warn(msg)

    # The max value in the data for each day of the year
    daily_max = np.empty(unique_days.size, data_arr.dtype)
    for i, day in enumerate(unique_days):
        daily_max[i] = np.max(data_arr[days == day])

    # Moving window max
    moving_window_max = aggregate_periodic(daily_max, halfwin, 'max')
    # Moving window mean
    moving_window_max_mean = aggregate_periodic(moving_window_max, halfwin)

    # Smooth ubc
    ubc = data_arr.copy()
    for day in unique_days:
        ubc[days == day] = moving_window_max_mean[unique_days == day]

    return ubc


def scale_by_upper_bound_climatology(data_arr, ubc, divide=True):
    """
    Scales all values in d using the annual cycle of upper bounds.

    Parameters
    ----------
    data_arr: np.Array
        Time series to be scaled. Is changed in-place.
    ubc: np.Array
        Upper bound climatology used for scaling.
    divide: boolean, optional
        If True then d is divided by upper_bound_climatology, otherwise they
        are multiplied.

    Returns
    -------
    new_arr: np.Array
        Time series array scaled to the upper bound climatology
    """
    # For each data observation we should have smooth ubc corresponding value
    assert data_arr.shape == ubc.shape, 'Data and ubc shapes differ'

    # If we are scaling to [0, 1]
    if divide:
        # Careful not to divide by zero. When upper bound is zero, set data to zero
        # (this is rather irrelevant, but only way ubc is zero is if data is zero there anyway)
        with np.errstate(divide='ignore', invalid='ignore'):
            new_arr = np.where(ubc == 0, 0, data_arr / ubc)
        # Make values within range
        new_arr[new_arr > 1] = 1
        new_arr[new_arr < 0] = 0

    # If we are scaling back from [0, 1]
    else:
        new_arr = data_arr * ubc
        # Make sure values don't exceed ubc
        new_arr[new_arr > ubc] = ubc[new_arr > ubc]

    return new_arr


def ccs_transfer_sim2obs_upper_bound_climatology(ubcs, days):
    """
    Transfers climatology trend observed in the simulated data to the observed data.

    Parameters
    ----------
    ubcs: dict
        np.Arrays of the upper bounds for each day in each time series
    days: dict
        np.Arrays with the day of year for each data point in each array

    Returns
    -------
    sim_fut_ba: np.Array
        Upper bound for the future bias adjusted data
    """
    # Must have the same coverage of days
    msg = f'Not all input data covers the same days. Check the calendar and/or missing values'
    assert np.all(np.unique(days['obs_hist']) == np.unique(days['sim_hist'])), msg
    assert np.all(np.unique(days['obs_hist']) == np.unique(days['sim_fut'])), msg

    # Data
    obs_hist = ubcs['obs_hist']
    sim_hist = ubcs['sim_hist']
    sim_fut = ubcs['sim_fut']

    # Unadjusted output of correct shape
    sim_fut_ba = sim_fut.copy()

    # For each day, find the ubc for adjustment. Should have same shape as sim_fut
    for day in np.unique(days['obs_hist']):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Get the ubc for the given day in each time series
            # Just need the value so take the first one
            ubc_sim_hist = sim_hist[days['sim_hist'] == day][0]
            ubc_sim_fut = sim_fut[days['sim_fut'] == day][0]
            ubc_obs_hist = obs_hist[days['obs_hist'] == day][0]
            # Calc the change factor
            change_factor = np.where(ubc_sim_hist == 0, 1, ubc_sim_fut / ubc_sim_hist)
            change_factor = np.maximum(0.1, np.minimum(10, change_factor))
            # Scale obs_hist by the change factor and save it into the correct spot in sim_fut_ba
            sim_fut_ba[days['sim_fut'] == day] = change_factor * ubc_obs_hist

    return sim_fut_ba


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
        Days to center windows around
    """
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


def get_data_in_window(window_center, data_loc, days, years, long_term_mean, params: Parameters):
    years_this_window = {}
    data_this_window = {}
    for key, data_arr in data_loc.items():
        # Get indices for data needed for this window
        m = window_indices_for_running_bias_adjustment(days[key], window_center, 31)
        # Associated year for each data point in the resulting data
        years_this_window[key] = years[key][m]
        # Sample invalid values
        replaced, invalid = sample_invalid_values(data_arr[m],
                                                  seed=1,
                                                  if_all_invalid_use=params.if_all_invalid_use,
                                                  warn=params.invalid_value_warnings)
        # The actual needed data in this window
        data_this_window[key] = replaced

    return data_this_window, years_this_window


def get_data_in_month(month, data_loc, years, month_numbers, long_term_mean, params: Parameters):
    years_this_month = {}
    data_this_month = {}
    for key, data_arr in data_loc.items():
        # Get indices for data needed for this window
        m = month_numbers[key] == month
        # Error if no data found for supplied month
        assert np.any(m), f'No data found for month {month} in {key}'
        # Associated year for each data point in the resulting data
        y = years[key]
        years_this_month[key] = None if y is None else y[m]
        # Sample invalid values
        replaced, invalid = sample_invalid_values(data_arr[m],
                                                  seed=1,
                                                  if_all_invalid_use=params.if_all_invalid_use,
                                                  warn=params.invalid_value_warnings)
        # The actual needed data in this window
        data_this_month[key] = replaced

    return data_this_month, years_this_month


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


def chunk_indexes(chunk_sizes):
    all_lat_indexes = np.arange(sum(chunk_sizes['lat']))
    all_lon_indexes = np.arange(sum(chunk_sizes['lon']))
    lat_indexes = {}
    lon_indexes = {}
    cum_sum = 0
    for i, size in enumerate(chunk_sizes['lat']):
        lat_indexes[i] = all_lat_indexes[np.arange(size) + cum_sum]
        cum_sum += size
    cum_sum = 0
    for i, size in enumerate(chunk_sizes['lon']):
        lon_indexes[i] = all_lon_indexes[np.arange(size) + cum_sum]
        cum_sum += size

    return lat_indexes, lon_indexes


def time_scraping(datasets: dict):
    """
    Function that turns the time variables for each input dataset into arrays
    of days of the year, month number, and year.

    Parameters
    __________
    datasets: dict
        Dictionary of xarray datasets

    Returns
    -------
    days: dict
        Dictionary of numpy arrays that give the day of year for each obs in each dataset
    month_numbers: dict
        Dictionary of numpy arrays that give the month number for each obs in each dataset
    years: dict
        Dictionary of numpy arrays that give the year for each obs in each dataset
    """
    # Empty output dictionaries
    days, month_numbers, years = {}, {}, {}

    # Iterate through each dataset in dictionary and get data
    for key, value in datasets.items():
        dates = pd.DatetimeIndex(value['time'].values)
        days[key] = dates.day_of_year
        month_numbers[key] = dates.month
        years[key] = dates.year

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
    d_replaced: Array
        Result of invalid data replacement.
    l_invalid: Array
        Boolean array indicating indices of replacement.

    """
    # make sure types and shapes of a and if_all_invalid_use fit
    space_shape = a.shape[1:]
    # if len(space_shape):
    #     msg = 'expected if_all_invalid_use to be an array'
    #     assert isinstance(if_all_invalid_use, np.ndarray), msg
    #     msg = 'shapes of a and if_all_invalid_use do not fit'
    #     assert if_all_invalid_use.shape == space_shape, msg
    # else:
    #     msg = 'expected if_all_invalid_use to be scalar'
    #     assert np.isscalar(if_all_invalid_use), msg

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
                d[j], seed, if_all_invalid_use, warn, l_invalid[j])
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


def subtract_or_add_trend(x, years, trend=None):
    """
    Subtracts or adds trend from or to x.

    Parameters
    ----------
    x: np.Array
        Time series.
    years: np.Array
        Years of time points of x used to subtract or add trend at annual
        temporal resolution.
    trend: array, optional
        Trend line. If provided then this is the trend line added to x.
        Otherwise, a trend line is computed and subtracted from x

    Returns
    -------
    y: np.Array
        Result of trend subtraction or addition from or to x.
    trend: np.Array, optional
        Trend line. Is only returned if the parameter trend is None.

    """
    assert x.size == years.size, 'size of x != size of years'
    unique_years = np.unique(years)

    # compute trend
    if trend is None:
        annual_means = np.array([np.mean(x[years == y]) for y in unique_years])
        r = sps.linregress(unique_years, annual_means)
        if r.pvalue < .05:  # detrend preserving multi-year mean value
            trend = r.slope * (unique_years - np.mean(unique_years))
        else:  # do not detrend because trend is insignificant
            trend = np.zeros(unique_years.size, dtype=x.dtype)
        return_trend = True
    else:
        msg = 'size of trend array != number of unique years'
        assert trend.size == unique_years.size, msg
        trend = -trend
        return_trend = False

    # subtract or add trend
    if np.any(trend):
        y = np.empty_like(x)
        for i, year in enumerate(unique_years):
            is_year = years == year
            y[is_year] = x[is_year] - trend[i]
    else:
        y = x.copy()

    # return result(s)
    if return_trend:
        return y, trend
    else:
        return y


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


def adjust_bias_one_month(data, years, params):
    # Detrend and randomize censored values
    # Saving future trend for adding back later
    trend_sim_fut = None
    for key, y in years.items():
        # detrending
        if params.detrend:
            data[key], t = subtract_or_add_trend(data[key], y)
            # Save future trend for adding back later
            if key == 'sim_fut':
                trend_sim_fut = t

        # Randomizing censored values (above/below thresholds)
        randomize_censored_values(data[key],
                                  params.lower_bound, params.lower_threshold,
                                  params.upper_bound, params.upper_threshold,
                                  True, False, 1)

    # TODO: Implement copula mbcn adjustment?

    y, unadjusted, non_standard = map_quantiles_parametric_trend_preserving(data, params)

    # re-introducing the trend
    if params.detrend:
        y = subtract_or_add_trend(y, years['sim_fut'], trend_sim_fut)

    # TODO: assert to infs or nans

    return y, unadjusted, non_standard


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
    i_sim_hist: np.Array
        indexes which inform where the simulated historical data is within the thresholds.
    i_sim_fut: np.Array
        indexes which inform where the simulated future data is within the thresholds.
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
    Generates pseudo future observation(s) by transferring a simulated climate
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

    # Compute x_obs_fut
    # Indexes for each type of bias
    i_neg_bias = x_sim_hist < x_obs_hist
    i_zero_bias = x_sim_hist == x_obs_hist
    i_pos_bias = x_sim_hist > x_obs_hist

    # If x_sim_fut < x_sim_hist < x_obs_hist or x_sim_fut > x_sim_hist > x_obs_hist
    i_additive = np.logical_or(
        np.logical_and(i_neg_bias, x_sim_fut < x_sim_hist),
        np.logical_and(i_pos_bias, x_sim_fut > x_sim_hist))

    # Empty result
    x_obs_fut = np.empty_like(x_obs_hist)

    # Ratio for negative bias case
    neg_ratio = (upper_bound - x_sim_fut[i_neg_bias]) / (upper_bound - x_sim_hist[i_neg_bias])
    # Values for negative bias case
    x_obs_fut[i_neg_bias] = upper_bound - (upper_bound - x_obs_hist[i_neg_bias]) * neg_ratio

    # Values for zero bias case
    x_obs_fut[i_zero_bias] = x_sim_fut[i_zero_bias]

    # Ratio for positive bias case
    pos_ratio = (x_sim_fut[i_pos_bias] - lower_bound) / (x_sim_hist[i_pos_bias] - lower_bound)
    # Values for positive bias case
    x_obs_fut[i_pos_bias] = lower_bound + (x_obs_hist[i_pos_bias] - lower_bound) * pos_ratio

    # Values for additive case
    x_obs_fut[i_additive] = x_obs_hist[i_additive] + x_sim_fut[i_additive] - x_sim_hist[i_additive]

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
        if p_lower_target > 0:
            lower_threshold_source = percentile1d(x_source, np.array([p_lower_target]))[0]
        elif not upper:
            lower_threshold_source = params.lower_bound
        else:
            lower_threshold_source = params.lower_bound - 1e-10 * (params.upper_bound - params.lower_bound)
        # make sure that lower_threshold_source < x_source
        # because otherwise sps.beta.ppf does not work
        i_lower = x_source <= lower_threshold_source
        i_source = np.logical_and(i_source, np.logical_not(i_lower))
        i_target = np.logical_and(i_target, x_target > params.lower_threshold)
        y[i_lower] = params.lower_bound
    if upper:
        if p_upper_target > 0:
            upper_threshold_source = percentile1d(x_source, np.array([1. - p_upper_target]))[0]
        elif not lower:
            upper_threshold_source = params.upper_bound
        else:
            upper_threshold_source = params.upper_bound + 1e-10 * (params.upper_bound - params.lower_bound)
        # make sure that x_source < upper_threshold_source
        # because otherwise sps.beta.ppf does not work
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
    x: np.Array
        Simulated time series.
    y: np.Array
        Observed time series.

    Returns
    -------
    z: np.Array
        Result of quantile mapping.
    non_standard: int
        0 when working normally, 1 when warning

    """
    if x.size == 0:
        msg = 'found no values in x: returning x'
        # TODO: Maybe have separate log file for warnings
        # warnings.warn(msg)
        return x, 1

    if np.unique(y).size < 2:
        msg = 'found fewer then 2 different values in y: returning x'
        # warnings.warn(msg)
        return x, 1

    p_x = (sps.rankdata(x) - 1.) / x.size  # percent points of x
    p_y = np.linspace(0., 1., y.size)  # percent points of sorted y
    z = np.interp(p_x, p_y, np.sort(y))  # quantile mapping
    return z, 0


def map_quantiles_non_parametric_with_constant_extrapolation(x, q_sim, q_obs):
    """
    Uses quantile-quantile pairs represented by values in q_sim and q_obs
    for quantile mapping of x.

    Values in x beyond the range of q_sim are mapped following the constant
    extrapolation approach, see Boe et al. (2007)
    <https://doi.org/10.1002/joc.1602>.

    Parameters
    ----------
    x : np.Array
        Simulated time series.
    q_sim : np.Array
        Simulated quantiles.
    q_obs : np.Array
        Observed quantiles.

    Returns
    -------
    y : np.Array
        Result of quantile mapping.

    """
    assert q_sim.size == q_obs.size
    ind_under = x < q_sim[0]
    ind_over = x > q_sim[-1]
    y = np.interp(x, q_sim, q_obs)
    y[ind_under] = x[ind_under] + (q_obs[0] - q_sim[0])
    y[ind_over] = x[ind_over] + (q_obs[-1] - q_sim[-1])
    return y


def map_quantiles_core(x_source, x_target, y, i_source, i_target, i_sim_fut, params):
    """
    Parameters:
    ----------
    x_source
    x_target
    y
    i_source
    i_target
    i_sim_fut
    params: basd.Parameters
        Parameters object containing details for the given run

    Returns:
    --------
    y
    unadjusted: int
        1 if values were left unadjusted
    non_standard: int
        1 if values were not fit as asked, but still adjusted
    """
    # Diagnostic values
    unadjusted = 0
    non_standard = 0

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
            x_source_map, non_standard = map_quantiles_non_parametric_brute_force(x_source[i_source], x_source_fit)
        else:
            x_source_map = x_source

        # Fix location and scale parameters
        floc = params.lower_threshold if lower else None
        fscale = params.upper_threshold - params.lower_threshold if lower and upper else None
        fwords = {'floc': floc, 'fscale': fscale}

        # Rice and weibull can't handle fscale parameter
        if params.distribution in ['rice', 'weibull']:
            fwords = {'floc': floc}

        # Fit distributions to x_source and x_target
        shape_loc_scale_source = fit(spsdotwhat, x_source_fit, fwords)
        shape_loc_scale_target = fit(spsdotwhat, x_target_fit, fwords)

        # This just uses MLE without fixing scale/location parameters ever
        # shape_loc_scale_source = spsdotwhat.fit(x_source_fit)
        # shape_loc_scale_target = spsdotwhat.fit(x_target_fit)

    # do non-parametric quantile mapping if needed
    if shape_loc_scale_source is None or shape_loc_scale_target is None:
        # This is the case when parametric was desired but failed
        if spsdotwhat is not None:
            msg = 'Parametric quantile mapping failed. Performing non-parametric quantile mapping instead.'
            non_standard = 1
            # TODO: Maybe write out to separate log file for warnings
            # warnings.warn(msg)
        p_zeroone = np.linspace(0., 1., params.n_quantiles + 1)
        q_source_fit = percentile1d(x_source_map, p_zeroone)
        q_target_fit = percentile1d(x_target_fit, p_zeroone)
        y[i_source] = map_quantiles_non_parametric_with_constant_extrapolation(x_source_map, q_source_fit, q_target_fit)

        # If doing non-parametric mapping, return here
        return y, unadjusted, non_standard

    # From here, assuming that parametric fit was successful
    # Here are the p-values of the simulated future values according to the
    # fit for the simulated future values
    p_source = spsdotwhat.cdf(x_source_map, *shape_loc_scale_source)
    # Limiting the p-values to not be too small or close to 1
    p_source = np.maximum(params.p_value_eps, np.minimum(1 - params.p_value_eps, p_source))

    # Compute target p-values
    if params.adjust_p_values:
        # TODO: Implement this case
        raise Exception('Adjusting p-values not yet implemented')
    else:
        p_target = p_source

    # ppf is inverse CDF function. Here we take the p-values of the target and translate
    # them to data according to the target distribution
    y[i_source] = spsdotwhat.ppf(p_target, *shape_loc_scale_target)

    return y, unadjusted, non_standard


def check_shape_loc_scale(spsdotwhat, shape_loc_scale):
    """
    Analyzes how distribution fitting has worked.

    Parameters
    ----------
    spsdotwhat : sps distribution class
        Known classes are [sps.norm, sps.weibull_min, sps.gamma, sps.rice,
        sps.beta].
    shape_loc_scale : tuple
        Fitted shape, location, and scale parameter values.

    Returns
    -------
    i : int
        0 if everything is fine,
        1 if there are infs or nans in shape_loc_scale,
        2 if at least one value in shape_loc_scale is out of bounds,
        3 if spsdotwhat is unknown.

    """
    # Return 1 if infs or nans
    if np.any(np.isnan(shape_loc_scale)) or np.any(np.isinf(shape_loc_scale)):
        return 1
    # Variance for normal dist must be positive
    elif spsdotwhat == sps.norm:
        return 2 if shape_loc_scale[1] <= 0 else 0
    # Weibull, gamma and rice distributions have positive parameter bounds
    elif spsdotwhat in [sps.weibull_min, sps.gamma, sps.rice]:
        return 2 if shape_loc_scale[0] <= 0 or shape_loc_scale[2] <= 0 else 0
    # Bounds for bet distribution
    elif spsdotwhat == sps.beta:
        return 2 if shape_loc_scale[0] <= 0 or shape_loc_scale[1] <= 0 \
                    or shape_loc_scale[0] > 1e10 or shape_loc_scale[1] > 1e10 else 0
    else:
        return 3


# TODO: Add option to try not fixing scale and location
# TODO: Return distribution parameters to somewhere accessible by user
def fit(spsdotwhat, x, fwords: dict):
    """
    Attempts to fit a distribution from the family defined through spsdotwhat
    to the data represented by x, holding parameters fixed according to fwords.

    A maximum likelihood estimation of distribution parameter values is tried
    first. If that fails the method of moments is tried for some distributions.

    Parameters
    ----------
    spsdotwhat : sps distribution class
        Known classes are [sps.norm, sps.weibull_min, sps.gamma, sps.rice,
        sps.beta].
    x: np.Array
        Data to be fitted.
    fwords: dict
        Optional location and scale parameters held fixed when fitting

    Returns
    -------
    shape_loc_scale: tuple
        Fitted shape, location, and scale parameter values if fitting worked,
        otherwise None.
    """
    # Try maximum likelihood estimation
    try:
        shape_loc_scale = spsdotwhat.fit(x, **fwords)
    except Exception as e:
        print(f'Exception: {e.__class__}, was unable to fit using MLE')
        shape_loc_scale = (np.nan,)

    # Try maximum likelihood estimation
    if check_shape_loc_scale(spsdotwhat, shape_loc_scale):
        msg = 'Maximum likelihood estimation failed. Trying method of moments.'
        if spsdotwhat == sps.gamma:
            msg += 'Method of moments'
            x_mean = np.mean(x) - fwords['floc']
            x_var = np.var(x)
            scale = x_var / x_mean
            shape = x_mean / scale
            shape_loc_scale = (shape, fwords['floc'], scale)
        elif spsdotwhat == sps.beta:
            msg += 'Method of moments'
            y = (x - fwords['floc']) / fwords['fscale']
            y_mean = np.mean(y)
            y_var = np.var(y)
            p = np.square(y_mean) * (1. - y_mean) / y_var - y_mean
            q = p * (1. - y_mean) / y_mean
            shape_loc_scale = (p, q, fwords['floc'], fwords['fscale'])

    else:
        msg = 'Maximum likelihood estimation succeeded.'

    # Check again
    if check_shape_loc_scale(spsdotwhat, shape_loc_scale):
        msg = 'Failed fit: returning None'
        warnings.warn(msg)
        return None
    elif msg != 'Maximum likelihood estimation succeeded.':
        msg += ' succeeded.'

    # do rough goodness of fit test to filter out worst fits using KS test
    ks_stat = sps.kstest(x, spsdotwhat.name, args=shape_loc_scale)[0]
    if ks_stat > .5:
        msg += ' Fit is not good: returning None.'
        warnings.warn(msg)
        return None

    return shape_loc_scale


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
    x_sim_fut = data['sim_fut']

    # Get data within thresholds and indexes of these observations
    data_within_thresholds, i_obs_hist, i_sim_hist, i_sim_fut = get_data_within_thresholds(data, params)

    # Non parametric quantile mapping
    # Use all values if unconditional_css_transfer
    if params.unconditional_ccs_transfer:
        x_target, unadjusted, non_standard = map_quantiles_non_parametric_trend_preserving(data, params)
    else:
        # use only values within thresholds
        x_target = x_obs_hist.copy()
        x_target[i_obs_hist], unadjusted, non_standard = map_quantiles_non_parametric_trend_preserving(data_within_thresholds, params)

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
    if not np.any(i_target) or not np.any(i_source):
        msg = 'Unable to do any quantile mapping' \
              + ': leaving %i value(s) unadjusted' % np.sum(i_source)
        # TODO: Perhaps write this to seperate log file for warnings
        unadjusted = 1
        # warnings.warn(msg)
        return y, unadjusted, 0

    # map quantiles
    result, unadjusted, non_standard = map_quantiles_core(x_source, x_target, y, i_source, i_target, i_sim_fut, params)

    return result, unadjusted, non_standard


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
    unadjusted: int
        Whether values were left anadusted (1 for yes, 0 for no)
    non_standard: int
        Whether adjustment had to tweak input parameters (1 for yes, 0 for no)
    """
    # Diagnostic values
    unadjusted = 0
    non_standard = 0

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
        # TODO: Maybe add separate log file for warnings
        # warnings.warn(msg)
        unadjusted = 1
        return y, unadjusted, non_standard
    elif n < params.n_quantiles + 1:
        msg = 'due to little input data: reducing n_quantiles to %i' % (n - 1)
        # TODO: Maybe add separate log file for warnings
        # warnings.warn(msg)
        non_standard = 1
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
    cdf_sim_fut_inv = np.interp(p, p_zeroone, q_sim_fut)
    cdf_sim_hist_inv = np.interp(p, p_zeroone, q_sim_hist)
    cdf_obs_hist_inv = np.interp(p, p_zeroone, q_obs_hist)

    # do augmented quantile delta mapping
    if params.trend_preservation == 'bounded':
        msg = 'lower_bound or upper_bound not specified'
        assert params.lower_bound is not None and params.upper_bound is not None, msg
        assert params.lower_bound < params.upper_bound, 'lower_bound >= upper_bound'
        y = ccs_transfer_sim2obs(
            cdf_obs_hist_inv, cdf_sim_hist_inv, cdf_sim_fut_inv,
            params.lower_bound, params.upper_bound)
    elif params.trend_preservation in ['mixed', 'multiplicative']:
        assert params.max_change_factor > 1, 'max_change_factor <= 1'
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(cdf_sim_hist_inv == 0, 1., cdf_sim_fut_inv / cdf_sim_hist_inv)
            y[y > params.max_change_factor] = params.max_change_factor
            y[y < 1. / params.max_change_factor] = 1. / params.max_change_factor
        y *= cdf_obs_hist_inv
        if params.trend_preservation == 'mixed':  # if not then we are done here
            assert params.max_adjustment_factor > 1, 'max_adjustment_factor <= 1'
            y_additive = cdf_obs_hist_inv + cdf_sim_fut_inv - cdf_sim_hist_inv
            fraction_multiplicative = np.zeros_like(y)
            fraction_multiplicative[cdf_sim_hist_inv >= cdf_obs_hist_inv] = 1.
            i_transition = np.logical_and(cdf_sim_hist_inv < cdf_obs_hist_inv,
                                          cdf_obs_hist_inv < params.max_adjustment_factor * cdf_sim_hist_inv)
            fraction_multiplicative[i_transition] = .5 * (1. +
                                                          np.cos((cdf_obs_hist_inv[i_transition] /
                                                                  cdf_sim_hist_inv[i_transition] - 1.) *
                                                                 np.pi / (params.max_adjustment_factor - 1.)))
            y = fraction_multiplicative * y + (1. -
                                               fraction_multiplicative) * y_additive
    elif params.trend_preservation == 'additive':
        y = cdf_obs_hist_inv + cdf_sim_fut_inv - cdf_sim_hist_inv
    else:
        msg = 'trend_preservation = ' + params.trend_preservation + ' not supported'
        raise AssertionError(msg)

    return y, unadjusted, non_standard
