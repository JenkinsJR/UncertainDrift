import numpy as np
from cartopy.geodesic import Geodesic


# the default geodesic instance is an ellipsoid representation of the Earth in
# metric units
EARTH = Geodesic()
# unit conversion
KM_TO_M = 1000


def random_sample(lon, lat, radius, n_samples, keepdims=True):
    """
    Randomly samples a circle around a point with a uniform distribution.

    Args:
        lon (float or ndarray): longitude.
        lat (float or ndarray): latitude.
        radius (float): radius of the circle.
        n_samples (int): number of samples.

    """
    points = np.column_stack([lon, lat])

    length = np.sqrt(np.random.rand(len(points), n_samples)) * radius
    theta = np.random.rand(len(points), n_samples) * 360

    all_samples = np.zeros((len(points), n_samples, 2))
    for i, (point, length, theta) in enumerate(zip(points, length, theta)):
        try:
            # earlier versions require access of .base
            samples = EARTH.direct(point, theta, length*KM_TO_M).base[:,:2]
        except TypeError:
            samples = EARTH.direct(point, theta, length*KM_TO_M)[:,:2]
        all_samples[i] = samples

    if keepdims:
        return all_samples
    else:
        return all_samples.squeeze()
