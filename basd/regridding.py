from rasterio.enums import Resampling

import rioxarray as rio
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
    sim_hist: xr.Dataset
        Simulated historical data
    sim_fut: xr.Dataset
        Simulated future data
    """
    # TODO: Allow coordinate system to be specified
    # Assert the coordinate reference system. Assumes CRS known by code ESPG:4326
    obs_hist.rio.write_crs(4326, inplace=True)
    sim_hist.rio.write_crs(4326, inplace=True)
    sim_fut.rio.write_crs(4326, inplace=True)

    # Temporarily renaming (lon, lat) --> (x, y) and ordering dimensions as time, y, x
    obs_hist_xy = obs_hist.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x')
    sim_hist_xy = sim_hist.rename({'lon': 'x', 'lat': 'y'}).transpose('time', 'y', 'x')

    # Project observational data onto simulated resolution
    obs_hist_resized_xy = obs_hist_xy.rio.reproject_match(sim_hist_xy,
                                                          resampling=Resampling.bilinear)

    # Revert coord named to lat, lon
    obs_hist_resized = obs_hist_resized_xy.rename({'x': 'lon', 'y': 'lat'})

    return obs_hist_resized
