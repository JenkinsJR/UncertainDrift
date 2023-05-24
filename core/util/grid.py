import numpy as np
from scipy import ndimage
from scipy import interpolate
from scipy import spatial


def mask_to_contour(mask, iterations=1):
    contour = mask ^ ndimage.binary_dilation(mask, iterations=iterations)

    return contour


def get_interpolator(a):
    return interpolate.interp1d(
        np.arange(len(a)), a, fill_value='extrapolate')


def arakawa_c_to_b(u, v, inplace=False):
    """
    Converts Arakawa C grid to B grid.
    
    Note that we approximate the operation by assuming a Cartesian grid rather
        than a curvlinear grid. The error is negligible for localised regions.

    Args:
        u (array): ndarray of C grid U values.
        v (array): ndarray of C grid V values.
        inplace (bool, optional): if True do operation inplace.
            Defaults to False.

    Returns:
        u (array): B grid U values.
        v (array): B grid V values.

    """
    if not inplace:
        u = u.copy()
        v = v.copy()
    
    # shift U up by 0.5px
    u *= 0.5
    u[..., :-1, :] += u[..., 1:, :]
    
    # shift V left by 0.5px
    v *= 0.5
    v[..., :, :-1] += v[..., :, 1:]
    
    if not inplace:
        return u, v


def flattened_meshgrid(x, y):
    return np.array(np.meshgrid(x, y)).reshape(2, -1)


def coordinate_bin_edges(a, n_bins=None):
    """
    Returns the bin edges of coordinate values.

    Args:
        a (array): 1D array of coordinate values.
        n_bins (int, optional): Number of bins to allocate. Defaults to the
            length of `a` plus one.

    Returns:
        (n_bins) array: array of interpolated bin edges.

    """
    if n_bins is None:
        n_bins = len(a)+1

    interp = get_interpolator(a)
    indices = np.linspace(-0.5, len(a)-1, n_bins)
    bin_edges = interp(indices)

    return bin_edges


def histogram(x_real, y_real, x_discrete, y_discrete):
    """
    Computes the 2D histogram of real-valued coordinates.

    Args:
        x_real (array): 1D array of real x coordinates.
        y_real (array): 1D array of real y coordinates.
        x_discrete (array): 1D array of discrete x coordinates.
        y_discrete (array): 1D array of discrete y coordinates.

    Returns:
        (y_real, x_real) array: array giving the density at each discrete
            position.
        (x_real) array: array giving the x coordinate bin edges.
        (x_real) array: array giving y coordinate bin edges.

    """
    x_edges = coordinate_bin_edges(x_discrete)
    y_edges = coordinate_bin_edges(y_discrete)

    return np.histogram2d(y_real, x_real, bins=[y_edges, x_edges])


def coords_to_indices(x_real, y_real, x_discrete, y_discrete):
    # find the (flattened) nearest neighbouring indices on the grid
    xs, ys = np.meshgrid(x_discrete, y_discrete)
    tree = spatial.cKDTree(np.c_[xs.ravel(), ys.ravel()])
    _, indices = tree.query(np.c_[x_real, y_real])

    return indices


def keep_from_mask(x_real, y_real, x_discrete, y_discrete, mask,
                   fill_nan=False):
    """
    Keeps only the real-valued coordinates that lie within the 2D mask.

    Args:
        x_real (array): 1D array of real x coordinates.
        y_real (array): 1D array of real y coordinates.
        x_discrete (array): 1D array of discrete x coordinates.
        y_discrete (array): 1D array of discrete x coordinates.
        mask (array): boolean mask of shape (y_discrete,x_discrete) giving the
            positions to keep.
        fill_nan (bool, optional): If True, set non-masked values to NaN.
            Defaults to False.

    """
    # convert real-valued coordinates to indices on the discrete grid
    indices = coords_to_indices(x_real, y_real, x_discrete, y_discrete)
    # lookup the indices to keep
    keep = mask.ravel()[indices]
    # remove the other values or set them to nan
    if fill_nan:
        x_real[~keep] = np.nan
        y_real[~keep] = np.nan
    else:
        x_real = x_real[keep]
        y_real = y_real[keep]

    return x_real, y_real


def masked_random_sample(mask, n_samples, x, y):
    """
    Randomly samples the discrete grid points on the mask.

    Args:
        mask (array): boolean mask of shape (y,x) giving the positions to
            include in the sampling.
        n_samples (int): approximate number of samples.
        x (array, optional): 1D array of x coordinate values.
        y (array, optional): 1D array of y coordinate values.

    Returns:
        (2, N) array: array giving the x and y sampling positions.

    """
    all_samples = flattened_meshgrid(x, y).T
    masked_samples = all_samples[mask.ravel()]

    indices = np.random.randint(0, len(masked_samples), n_samples)
    random_samples = masked_samples[indices].T

    return random_samples
