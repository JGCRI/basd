from rasterio.enums import Resampling

import xarray as xr


# TODO: Eventually we will want to look into methods using xESMF package based on the ESMF
#   tools. But we should always include the option to use rioxarray, as xESMF is not compatible
#   with windows OS.
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
    # TODO: Allow coordinate system to be specified
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
    resolution. Effectivley the opposite of the match_grids function

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
    obs_fine.rio.write_crs(4326, inplace=True)
    sim_coarse.rio.write_crs(4326, inplace=True)

    # Temporarily renaming (lon, lat) --> (x, y) and ordering dimensions as time, y, x
    obs_fine_xy = obs_fine.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x', ...)
    sim_coarse_xy = sim_coarse.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x', ...)

    # Project observational data onto simulated resolution
    sim_fine_xy = sim_coarse_xy.rio.reproject_match(obs_fine_xy,
                                                    resampling=Resampling.bilinear)

    # Revert coord named to lat, lon
    sim_fine = sim_fine_xy.rename({'x': 'lon', 'y': 'lat'})

    return sim_fine