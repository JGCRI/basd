from rasterio.enums import Resampling
from functools import reduce
from math import sqrt

import numpy as np
import xarray as xr
import xesmf


def factors(n):
    step = 2 if n % 2 else 1
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(sqrt(n)) + 1, step) if n % i == 0)))


def match_grids(obs_hist: xr.Dataset, sim_hist: xr.Dataset, sim_fut: xr.Dataset):
    """
    Function which remaps the observational data to the simulated data spatial resolution

    Parameters
    ----------
    obs_hist: xr.Dataset
        Observational data
    sim_hist: xr.Dataset
        Simulated historical data
    sim_fut: xr.Dataset
        Simulated future data

    Returns
    -------
    obs_hist_resized: xr.Dataset
        Observational data resized to match simulated data
    """
    # Assert the coordinate reference system. Assumes CRS known by code ESPG:4326
    obs_hist.rio.write_crs(4326, inplace=True)
    sim_hist.rio.write_crs(4326, inplace=True)
    sim_fut.rio.write_crs(4326, inplace=True)

    # Temporarily renaming (lon, lat) --> (x, y) and ordering dimensions as time, y, x
    obs_hist_xy = obs_hist.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x', ...)
    sim_hist_xy = sim_hist.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x', ...)

    # Project observational data onto simulated resolution
    obs_hist_resized_xy = obs_hist_xy.rio.reproject_match(sim_hist_xy,
                                                          resampling=Resampling.bilinear)

    # Revert coord named to lat, lon
    obs_hist_resized = obs_hist_resized_xy.rename({'x': 'lon', 'y': 'lat'})

    return obs_hist_resized


def interpolate_for_downscaling(obs_fine: xr.Dataset, sim_coarse: xr.Dataset):
    """
    Function for interpolating the simulated coarse data to the finer observational data
    resolution. Effectively the opposite of the match_grids function

    Parameters
    ----------
    obs_fine: xr.Dataset
        Observational data at fine resolution
    sim_coarse: xr.Dataset
        Simulated data at coarse resolution

    Returns
    -------
    sim_fine: xr.Dataset
        Observational data resized to match simulated data
    """
    # Assert the coordinate reference system. Assumes CRS known by code ESPG:4326
    obs_fine = obs_fine.rio.write_crs(4326)
    sim_coarse = sim_coarse.rio.write_crs(4326)

    # Temporarily renaming (lon, lat) --> (x, y) and ordering dimensions as time, y, x
    obs_fine_xy = obs_fine.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x', ...)
    sim_coarse_xy = sim_coarse.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x', ...)

    # Project observational data onto simulated resolution
    sim_fine_xy = sim_coarse_xy.rio.reproject_match(obs_fine_xy,
                                                    resampling=Resampling.bilinear)

    # Revert coord named to lat, lon
    sim_fine = sim_fine_xy.rename({'x': 'lon', 'y': 'lat'})

    return sim_fine


def reproject_for_integer_factors(obs_fine: xr.Dataset, sim_coarse: xr.Dataset, variable: str, periodic: bool = True):
    """
    Re-projects grids if necessary so that downscaling factors are positive integers

    Parameters
    ----------
    obs_fine: xr.Dataset
        Observational data at fine resolution
    sim_coarse: xr.Dataset
        Simulated data at coarse resolution
    variable: str
        The variable to be reprojected

    Returns
    -------
    sim_coarse: xr.Dataset
        Simulated data at coarse resolution, now an even factor or obs_fine
    """
    # Coordinate sequences
    fine_lats = obs_fine.coords['lat'].values
    fine_lons = obs_fine.coords['lon'].values
    coarse_lats = sim_coarse.coords['lat'].values
    coarse_lons = sim_coarse.coords['lon'].values

    # Get the downscaling factors
    f_lat = len(fine_lats) / len(coarse_lats)
    f_lon = len(fine_lons) / len(coarse_lons)

    # If integers already, return
    if f_lat.is_integer() & f_lon.is_integer():
        return sim_coarse

    # Else, get the nearest integer factor that divides the grid evenly
    # TODO: Broken for grids that are already less than 1 degree. Currently always downscales,
    #       which currently leads to coarse grid being same as fine grid
    lat_facts = factors(len(fine_lats))
    lon_facts = factors(len(fine_lons))
    # If we are thinking about downscaling to the observational dataset, don't. Instead, upscale.
    if(f_lat > 1 and f_lat < 2):
        f_lat = np.sort([x for x in lat_facts if x > 1])[0]
    # Generally though, tend to downscale
    else:
        f_lat = np.sort([x for x in lat_facts if x <= f_lat])[-1]
    # Same as above, but for longitudinal scaling
    if(f_lon > 1 and f_lon < 2):
        f_lon = np.sort([x for x in lon_facts if x > 1])[0]
    else:
        f_lon = np.sort([x for x in lon_facts if x <= f_lon])[-1]

    # Get template dataset with the right lat/lon series by coarsening fine dataset by the found factors
    # template = obs_fine.copy()
    template = xr.Dataset(coords={'lat': fine_lats, 'lon': fine_lons}).coarsen(lat=2, lon=2).sum()

    # Do the regridding and save as new coarse dataset
    coarse_to_finer_regridder = xesmf.Regridder(sim_coarse, template, 'bilinear', periodic=periodic)
    sim_coarse_arr = coarse_to_finer_regridder(sim_coarse[variable])
    sim_coarse = sim_coarse_arr.to_dataset(name=variable)

    return sim_coarse


def project_onto(to_project: xr.Dataset, template: xr.Dataset, variable: str, periodic: bool = True):
    """
    Reprojects one dataset to have the same coordinates of the template.

    Parameters
    ----------
    to_project: xr.Dataset
        xarray Dataset which we wish to change resolution/coordinates
    template: xr.Dataset
        xarray Dataset whose coordinates are being used as a template to match to
    variable: str
        Name of the dataset variable of interest
    periodic: bool, True
        Whether longitude coordinates are periodic

    Returns
    -------
    projected_dataset: xr.Dataset
        xarray Dataset which is the original dataset to project,
        bilinearly interpolated to match the coordinates of the template data
    """

    # Get the lat/lon coordinate sequences as template
    template_coords = xr.Dataset(coords={'lat': template.lat, 'lon': template.lon})

    # Create regridding object from xesmf
    regridder = xesmf.Regridder(to_project, template_coords, 'bilinear', periodic=periodic)

    # Do the regridding, result is a xr.DataArray
    projected_array = regridder(to_project[variable])

    # Create new dataset object
    projected_dataset = projected_array.to_dataset(name=variable)
    # projected_dataset = xr.Dataset({variable: projected_array})

    return projected_dataset
