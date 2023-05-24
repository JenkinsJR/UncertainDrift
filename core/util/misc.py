def yield_n(iter, n):
    """
    Returns the next n elements from an iterator.

    Args:
        iter (iterator): Iterator.
        n (int): Number of elements.

    """
    return [next(iter) for _ in range(n)]


def interpolated(v1, v2, x):
    """Returns the interpolation of two values.

    Args:
        v1 (float): First value
        v2 (float): Second value.
        x (float): A number in the range of 0-1 used to interpolate between the
            two given values.

    """
    return (1 - x) * v1 + x * v2