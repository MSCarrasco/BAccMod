"""
Define the background model function and fitting method.
"""

import numpy as np


def log_factorial(count_map):
    """
    Returns the log of the factorial of elements of `count_map` while computing eah value only once.
    Parameters
    ----------
    count_map: Array-like of integers
        Input for which we want the factorial of all elements

    Returns
    -------
        The factorial of count_map in log scale

    """
    max_input = np.max(count_map)
    all_int = np.arange(0, max_input + 1)
    all_int[0] = 1
    log_int = np.log(all_int)
    log_int_factorials = np.cumsum(log_int)
    log_factorial_count_map = log_int_factorials[count_map]
    return log_factorial_count_map


def log_poisson(x, mu, log_factorial_x):
    return -mu + x * np.log(mu) - log_factorial_x


def gaussian2d(x, y, size, x_cm, y_cm, width, length, psi, **kwargs):
    """
    Evaluate the bi-dimensional gaussian law.

    Parameters
    ----------
    x, y: float 1D array
        Position at which the log gaussian is evaluated
    size: float
        Integral of the 2D Gaussian at the provided coordinates
    x_cm, y_cm: float
        Center of the 2D Gaussian
    width, length: float
        Standard deviations of the 2 dimensions of the 2D Gaussian law
    psi: float
        Orientation of the 2D Gaussian

    Returns
    -------
    gauss2d: float 1D array
        Evaluation of the 2D gaussian law at (x,y)

    """
    # Compute the x and y coordinates projection in the 2D gaussian length and width coordinates
    le = (x - x_cm) * np.cos(psi) + (y - y_cm) * np.sin(psi)
    wi = -(x - x_cm) * np.sin(psi) + (y - y_cm) * np.cos(psi)
    a = 2 * length ** 2
    b = 2 * width ** 2
    # Evaluate the 2D gaussian term
    gauss2d = np.exp(-(le ** 2 / a + wi ** 2 / b))
    gauss2d = size / np.sum(gauss2d) * gauss2d
    return gauss2d


def gaussian1d(x, y, size, x_cm, y_cm, width, **kwargs):
    """
    Evaluate the bi-dimensional gaussian law assuming a single standard deviation.
    Thus removing the `length` and `psi` from `gaussian2d`.

    Parameters
    ----------
    x, y: float 1D array
        Position at which the log gaussian is evaluated
    size: float
        Integral of the 2D Gaussian
    x_cm, y_cm: float
        Center of the 2D Gaussian
    width: float
        Standard deviations of the 2 dimensions of the 2D Gaussian law

    Returns
    -------
    gauss2d: float 1D array
        Evaluation of the 2D gaussian law at (x,y)

    """
    # Compute the x and y coordinates projection in the 2D gaussian length and width coordinates
    le = (x - x_cm)
    wi = (y - y_cm)
    a = 2 * width ** 2
    # Evaluate the 2D gaussian term
    gauss2d = np.exp(-(le ** 2 + wi ** 2) / a)
    gauss2d = size / np.sum(gauss2d) * gauss2d
    return gauss2d


def ylinear_1dgaussian(x, y, size, x_cm, y_cm, width, y_gradient, **kwargs):
    """
    Adds a linear gradient to `1dgaussian`
    Parameters
    ----------
    x, y, size, x_cm, y_cm, width: see `1dgaussian`
    y_gradient: float

    """
    return (1 + y * y_gradient) * gaussian1d(x, y, size, x_cm, y_cm, width)


def bilinear_1dgaussian(x, y, size, x_cm, y_cm, width, x_gradient, y_gradient, **kwargs):
    """
    Adds a linear gradient to `1dgaussian`
    Parameters
    ----------
    x, y, size, x_cm, y_cm, width: see `1dgaussian`
    x_gradient: float
    y_gradient: float

    """
    return (1 + x * x_gradient) * (1 + y * y_gradient) * gaussian1d(x, y, size, x_cm, y_cm, width)


def fit_seed(corrected_counts, x, method):
    seeds, bounds = None, None
    size = np.sum(corrected_counts)
    w = max(x)
    if method == 'fit_gaussian':
        seeds = {
            'size': size,
            'x_cm': 0,
            'y_cm': 0,
            'width': w / 2,
            'length': w / 2,
            'psi': 0
        }
        bounds = {
            'size': (0, 10 * size),
            'x_cm': (-w, w),
            'y_cm': (-w, w),
            'width': (w / len(x), 2 * w),
            'length': (w / len(x), 2 * w),
            'psi': (0, 2 * np.pi)
        }
    if method == 'fit_xlin_gaussian':
        seeds = {
            'size': size,
            'x_cm': 0,
            'y_cm': 0,
            'width': w / 2,
            'y_gradient': 0
        }
        bounds = {
            'size': (0, 10 * size),
            'x_cm': (-w, w),
            'y_cm': (-w, w),
            'width': (w / len(x), 2 * w),
            'y_gradient': (-w, w)
        }
    if method == 'fit_bilin_gaussian':
        seeds = {
            'size': size,
            'x_cm': 0,
            'y_cm': 0,
            'width': w / 2,
            'x_gradient': 0,
            'y_gradient': 0
        }
        bounds = {
            'size': (0, 10 * size),
            'x_cm': (-w, w),
            'y_cm': (-w, w),
            'width': (w / len(x), 2 * w),
            'x_gradient': (-w, w),
            'y_gradient': (-w, w)
        }
    return seeds, bounds
