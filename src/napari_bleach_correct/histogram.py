from typing import Tuple

import numpy as np


def histogram_correct(
        images: np.ndarray,
        contrast_limits: Tuple[int, int],
        match: str = "first"
) -> np.ndarray:
    """
    Bleaching correction by matching histograms to a reference image.

    The correct pixel values can be calculated by the cumulative distribution function
    of a frame and its reference frame.

    .. math::
        p' = CDF_{ref}^{-1}(CDF_i(p))

    References:
        Miura K. Bleach correction ImageJ plugin for compensating the photobleaching of
        time-lapse sequences. F1000Res. 2020 Dec 21;9:1494.
        doi: 10.12688/f1000research.27171.1

    :param images: Image stack of shape `N, H, W`.
    :param contrast_limits: Lower and upper intensity bound.
    :param match: Match frame histogram with 'first' our 'neighbor' histogram.
    """
    # cache image dtype
    dtype = images.dtype
    k, m, n = images.shape

    assert (
            len(images.shape) == 3
    ), f"Expected 3d image stack, instead got {len(images.shape)} dimensions"

    avail_match_methods = ["first", "neighbor"]
    assert (
        match in avail_match_methods
    ), f"'match' expected to be one of {avail_match_methods}, instead got {match}"

    # flatten the last dimensions and calculate normalized cdf
    images = images.reshape(k, -1)
    values, cdfs = [], []

    for i in range(k):

        if i > 0:
            if match == "first":
                match_ix = 0
            else:
                match_ix = i - 1

            val, ix, cnt = np.unique(images[i, ...].flatten(), return_inverse=True, return_counts=True)
            cdf = np.cumsum(cnt) / (m * n)

            interpolated = np.interp(cdf, cdfs[match_ix], values[match_ix])
            images[i, ...] = interpolated[ix]

        if i == 0 or match == "neighbor":
            val, cnt = np.unique(images[i, ...].flatten(), return_counts=True)
            cdf = np.cumsum(cnt) / (m * n)
            values.append(val)
            cdfs.append(cdf)

    images = images.reshape(k, m, n)
    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)
