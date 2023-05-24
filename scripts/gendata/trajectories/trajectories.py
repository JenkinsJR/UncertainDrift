import numpy as np
import xarray as xr
from pathlib import Path

import parcels.kernels

from core import util
from core.data import simulate
from core.data import generate


def _fetch_kernel(name, fieldset, c_include):
    # lookup kernel func from name in both parcels/local kernel modules
    d = parcels.kernels.__dict__.copy()
    d.update(simulate.kernels.__dict__)
    kernel_func = d[name]
    
    # initialise a dummy pset in order to initialise the kernel
    # (required to ensure fieldset has the attribute UV)
    pset = parcels.ParticleSet(fieldset)
    # create kernel with included c code
    kernel = pset.Kernel(kernel_func, c_include=c_include, delete_cfiles=False)
    
    return kernel


def _init_particles(lon, lat, mesh, n_particles, radius):
    # generate an ensemble of particles for each position
    particles = generate.radius_ensemble(
        lon, lat, mesh.lon, mesh.lat, mesh.mask, n_particles, radius)

    # flatten particle info and assign each ensemble an id
    ensemble_ids = np.repeat(np.arange(particles.shape[0]), particles.shape[1])
    particles = np.vstack((particles.reshape(-1, 2).T, ensemble_ids))

    # remove NaN particles
    return particles[:, ~np.isnan(particles[0])]


def save_trajectories(directory, seeds, fieldset, particle_runtime,
                      particle_output_dt, particle_dt, kernel_name, kernel_path,
                      mesh, n_particles, ensemble_radius, start=0, end=None,
                      compile_only=False, c_include=''):
    kernel = _fetch_kernel(kernel_name, fieldset, c_include)
    kernel_path = kernel_path.with_suffix('.so')
    kernel_path.parent.mkdir(exist_ok=True)

    save_paths = util.path.gen_numerical(
        directory, '.nc', mkdir=(not compile_only))
    for i, (save_path, time) in enumerate(zip(save_paths, seeds)):
        if (start is not None and i < start) or (end is not None and i > end):
            continue

        lon, lat, ensemble_ids = _init_particles(
            *seeds[time], mesh, n_particles, ensemble_radius)

        simulate.trajectories(
            save_path, kernel_path, fieldset, lon, lat, time, ensemble_ids,
            particle_runtime, particle_output_dt, particle_dt,
            *mesh.boundary_coordinates, kernel=kernel,
            compile_only=compile_only)