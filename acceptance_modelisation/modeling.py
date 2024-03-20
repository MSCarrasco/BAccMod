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


def ring_bi_gaussian(x, y, size, r_cm, width, **kwargs):
    r = np.sqrt(x ** 2 + y ** 2)
    a = 2 * width ** 2
    # Evaluate the 2D gaussian term
    gauss2d_1 = np.exp(-(r - r_cm) ** 2 / a)
    gauss2d_2 = np.exp(-(-r - r_cm) ** 2 / a)
    gauss2d = gauss2d_1 + gauss2d_2
    gauss2d = size / np.sum(gauss2d) * gauss2d
    return gauss2d


def bi_gaussian(x, y, size, ratio, width, width2, **kwargs):
    r2 = x ** 2 + y ** 2
    a = 2 * width ** 2
    a2 = 2 * width2 ** 2
    # Evaluate the 2D gaussian term
    gauss2d_1 = np.exp(-r2 / a)
    gauss2d_2 = ratio * np.exp(-r2 / a2)
    gauss2d = gauss2d_1 + gauss2d_2
    gauss2d = size / np.sum(gauss2d) * gauss2d
    return gauss2d

def bilin_bi_gaussian(x, y, size, ratio, width, width2, x_gradient, y_gradient, **kwargs):
    gauss2d = (1 + x * x_gradient) * (1 + y * y_gradient) * bi_gaussian(x, y, size, ratio, width, width2)
    gauss2d = size / np.sum(gauss2d) * gauss2d
    return gauss2d

def radial_poly(x, y, size, p0, p1, p2, p3, p4, p5):
    r = np.sqrt(x ** 2 + y ** 2)
    poly = p0 + p1 * r + p2 * r ** 2 + p3 * r ** 3 * p4 * r ** 4 + p5 * r ** 5
    return size / np.sum(poly) * poly

def cauchy(x, y, size, a):
    r2 = x ** 2 + y ** 2
    ca = 1/(r2 + a**2)
    return size / np.sum(ca) * ca

def bicauchy(x, y, size, a, r0):
    r = np.sqrt(x ** 2 + y ** 2)
    ca = 1/((r-r0)**2 + a**2) + 1/((-r-r0)**2 + a**2)
    return size / np.sum(ca) * ca

def mod_tanh(x, y, size, speed, r0):
    r = np.sqrt(x ** 2 + y ** 2)
    th = np.tanh(r0-r*speed) + 1
    return size / np.sum(th) * th

def fit_seed(corrected_counts, x, method):
    seeds, bounds = None, None
    size = np.sum(corrected_counts)
    w = max(x)
    if method == 'tanh':
        seeds = {
            'size': size,
            'speed': 1,
            'r0': 2
        }
        bounds = {
            'size': (0, 10 * size),
            'speed': (0.1 * w, 5 * w),
            'r0': (0.1 * w, 5 * w)
        }
    if method == 'cauchy':
        seeds = {
            'size': size,
            'a': 1,
        }
        bounds = {
            'size': (0, 10 * size),
            'a': (0.1 * w, 5 * w)
        }
    if method == 'bicauchy':
        seeds = {
            'size': size,
            'a': 1,
            'r0':2
        }
        bounds = {
            'size': (0, 10 * size),
            'a': (0.1 * w, 5 * w),
            'r0': (0.1 * w, 5 * w)
        }
    if method == 'gaussian':
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
    if method == 'xlin_gaussian':
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
    if method == 'bilin_gaussian':
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
    if method == 'ring_bi_gaussian':
        seeds = {
            'size': size,
            'r_cm': 0,
            'width': w / 2
        }
        bounds = {
            'size': (0, 10 * size),
            'r_cm': (-w, w),
            'width': (w / len(x), 2 * w)
        }
    if method == 'bi_gaussian':
        seeds = {
            'size': size,
            'ratio': 1,
            'width': w / 2,
            'width2': w
        }
        bounds = {
            'size': (0, 10 * size),
            'ratio': (0.01, 10),
            'width': (w / len(x), 2 * w),
            'width2': (w / len(x), 2 * w)
        }
    if method == 'bilin_bi_gaussian':
        seeds = {
            'size': size,
            'ratio': 1,
            'width': w / 2,
            'width2': w,
            'x_gradient': 0,
            'y_gradient': 0
        }
        bounds = {
            'size': (0, 10 * size),
            'ratio': (0.01, 10),
            'width': (w / len(x), 2 * w),
            'width2': (w / len(x), 2 * w),
            'x_gradient': (-w, w),
            'y_gradient': (-w, w)
        }
    if method == 'radial_poly':
        seeds = {
            'size': size,
            'p0': 1,
            'p1': 0,
            'p2': 0,
            'p3': 0,
            'p4': 0,
            'p5': 0
        }
        bounds = {
            'size': (0, 10 * size),
            'p0': (0, 2),
            'p1': (-5*w, 5*w),
            'p2': (-5*w, 5*w),
            'p3': (-5*w, 5*w),
            'p4': (-5*w, 5*w),
            'p5': (-5*w, 5*w)
        }
    return seeds, bounds
