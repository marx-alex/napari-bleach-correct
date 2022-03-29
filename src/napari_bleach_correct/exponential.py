from typing import Tuple
import logging

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def exp(x, a, b):
    return a * np.exp(-b * x)


def bi_exp(x, a, b, c, d):
    return (a * np.exp(-b * x)) + (c * np.exp(-d * x))


def exponential_correct(
        images: np.ndarray,
        contrast_limits: Tuple[int, int],
        method: str = "mono"
) -> np.ndarray:
    """
    Drift estimation of fluorescence signal by fitting the mean intensity to an exponential curve.
    The image is corrected by the decay in the normalized exponential function.
    The time intervals between frames should be equal.

    A mono- or a bi-exponential function can be used.
    Mono-exponential function:
    .. math::
        \bar{I}'_i(x,y) = a\euler^{-bi}

    Bi-exponential function:
    .. math::
        \bar{I}'_i(x,y) = a\euler^{-bi} + c\euler^{-di}

    :param images: Image stack of shape `N, H, W`.
    :param contrast_limits: Lower and upper intensity bound.
    :param method: Type of exponential curve ("mono" or "bi").
    """
    # cache image dtype
    dtype = images.dtype

    # scale image between 0 and 1
    images = images / contrast_limits[1]

    assert (
            len(images.shape) == 3
    ), f"Expected 3d image stack, instead got {len(images.shape)} dimensions"

    # choose exponential curve
    avail_methods = ["mono", "bi"]
    if method == "mono":
        func = exp
    elif method == "bi":
        func = bi_exp
    else:
        raise NotImplementedError(
            f"method must be one of {avail_methods}, instead got {method}"
        )

    # calculate the mean intensity for every frame
    # store the intensity from the first frame
    I_mean = np.mean(images, axis=(1, 2))

    # fit curve
    x_data = np.arange(images.shape[0])
    with np.errstate(over="ignore"):
        try:
            popt, _ = curve_fit(func, x_data, I_mean)
            # get theoretical values
            f_ = np.vectorize(func)(x_data, *popt)
        except (ValueError, RuntimeError, Warning):
            f_ = np.ones(x_data.shape)

    # calculate r squared
    residuals = I_mean - f_
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((I_mean - np.mean(I_mean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    logger.info(f"R-squared value for fitting a {method}-exponential curve: {r_squared}")

    # normalize theoretical data
    f = f_ / np.max(f_)

    # multiply every frame by its ratio
    images = images / f.reshape(-1, 1, 1)

    # rescale and avoid overflow
    images = images * contrast_limits[1]
    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)
