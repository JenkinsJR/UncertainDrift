import numpy as np
import xarray as xr

from core.data import generate
from core import util


def save_seeds(save_path, mesh, temporal_positions, n_samples):
    # randomly sample temporal positions
    times = np.random.choice(temporal_positions, n_samples)
    times.sort()

    # for each temporal position, randomly sample the spatial positions
    seeds = {time: generate.particle_seeds(
                mesh.mask, mesh.lon, mesh.lat, count)
             for time, count in zip(*np.unique(times, return_counts=True))}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, seeds)