import numpy as np


def split(elements, proportions):
    """
    Randomly splits elements into groups of given proportions.
    
    Args:
        elements (int or array_like): Sequence (or number) of elements.
        proportions (array_like): Sequence of proportions that sum to 1.

    """
    proportions = np.array(proportions)
    assert np.isclose(proportions.sum(), 1)
    proportions = proportions[:-1]
    
    if type(elements) is int:
        elements = np.arange(elements)
    elements = np.random.permutation(elements)
    
    splits = np.round(proportions*len(elements)).cumsum().astype(int)
    
    return np.split(elements, splits)