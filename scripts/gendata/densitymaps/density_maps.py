import numpy as np
import xarray as xr
from tqdm import tqdm

from core.data import generate
from core import util


def save_density_maps(directory, mesh, all_trajectories, sigma):
    grid_shape = len(mesh.lat), len(mesh.lon)

    save_paths = util.path.gen_numerical(directory, '.nc')
    # time by time, iterate through all of the saved positions and trajectories
    for i, (save_path, trajectories) in enumerate(zip(
            save_paths, all_trajectories)):
        # not all ensembles have the same particle counts, so split the data
        # into their respective ensembles by computing the split points
        _, ensemble_sizes = np.unique(
            trajectories.ensemble_id, return_counts=True)
        ensemble_splits = ensemble_sizes[:-1].cumsum().data
        lon_data = np.split(trajectories.lon.data, ensemble_splits, axis=1)
        lat_data = np.split(trajectories.lat.data, ensemble_splits, axis=1)

        # initialise memory for all density maps at the current time sample
        n_ensembles = len(lon_data)
        obs_times = trajectories.time.data
        n_observations = len(obs_times)
        density_maps = np.zeros((n_ensembles, n_observations, *grid_shape),
                                dtype='float32')

        # iterate through all ensembles
        for j, (ensemble_size, lon, lat) in enumerate(
                zip(ensemble_sizes, lon_data, lat_data)):
            # iterate through all observations
            for obs in range(n_observations):
                # skip if data is all NaNs
                if np.isnan(lon[obs]).all():
                    continue
                density_maps[j, obs] = generate.density_map(
                    lon[obs], lat[obs], mesh.lon, mesh.lat,
                    sigma=sigma, n_points=ensemble_size.item())
                
        ds = xr.Dataset(
            {
                'density_map': (['ensemble_id', 'obs', 'lat', 'lon'],
                                density_maps)
            },
            coords={
                'time': ('obs', obs_times),
                'lat': mesh.lat,
                'lon': mesh.lon
            }
        )
        ds.to_netcdf(save_path)