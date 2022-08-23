import numpy as np
import xarray
from netCDF4 import num2date


def ma2a(a, raise_error: bool):
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
