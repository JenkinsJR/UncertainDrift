from scipy.ndimage import gaussian_filter

from core.util import grid


def particle_seeds(ocean_mask, lon_grid, lat_grid, n_samples):
    """
    Randomly samples a set of ensembles and their particle seeds.

    Args:
        ocean_mask (ndarray): 2D mask of ocean samples.
        lon_grid (ndarray): 1D array of ocean_mask longitude coordinates.
        lat_grid (ndarray): 1D array of ocean_mask latitude coordinates.
        n_samples (int): Number of ensembles.

    Returns:
        initial_positions (ndarray of shape (n_samples,)): The initial
            positions corresponding to axis 0 of particle_seeds.

    """
    initial_pos_mask = ocean_mask

    # randomly sample the (non-overlapping) initial positions
    initial_positions = grid.masked_random_sample(
        initial_pos_mask, n_samples, lon_grid, lat_grid)

    return initial_positions


def radius_ensemble(lon, lat, lon_grid, lat_grid, ocean_mask, n_particles,
                    radius):
    from core.util import geodesic

    # generate a set of random pertubations for each initial position
    samples = geodesic.random_sample(lon, lat, radius, n_particles)
    # set values outside of the ocean mask to nan
    flattened_samples = samples.reshape(-1, samples.shape[-1]).T
    grid.keep_from_mask(*flattened_samples, lon_grid, lat_grid, ocean_mask,
                        fill_nan=True)

    return samples


def density_map(lon, lat, lon_grid, lat_grid, sigma=0, normed=True,
                n_points=None):
    """
    Estimates the density map over the 2D grid using a smoothed histogram.

    Args:
        lon (ndarray): 1D array of lon values.
        lat (ndarray): 1D array of lat values.
        lon_grid (ndarray): 1D array of discrete lon grid coordinates.
        lat_grid (ndarray): 1D array of discrete lat grid coordinates.
        sigma (float, optional): Gaussian smoothing sigma. Defaults to 0.
        normed (bool, optional): If True, the returned density map sums to 1.
        n_points (int, optional): If given, use this value to normalise the
            densities instead of the total count.

    """
    density = grid.histogram(lon, lat, lon_grid, lat_grid)[0]
    if sigma > 0:
        gaussian_filter(density, sigma, output=density)

    # normalise the density range
    if normed:
        if n_points is None:
            n_points = density.sum()
        density /= n_points

    return density