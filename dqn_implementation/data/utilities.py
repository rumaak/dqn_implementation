import numpy as np


def rgb2grayscale(array_rgb):
    """
    Expects 3d numpy array with (R, G, B) as last dimension (eg 210x160x3)
    """
    consts = np.array([0.2989, 0.5870, 0.1140])
    array_grayscale = np.sum(array_rgb * consts[None, None, :], axis=-1)
    return array_grayscale
