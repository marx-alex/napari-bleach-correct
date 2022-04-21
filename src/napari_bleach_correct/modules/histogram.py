from typing import Tuple

import numpy as np
from napari.types import ImageData


def histogram_correct(
        images: ImageData,
        contrast_limits: Tuple[int, int],
        match: str = "first"
) -> ImageData:
    # cache image dtype
    dtype = images.dtype

    assert (
            3 <= len(images.shape) <= 4
    ), f"Expected 3d or 4d image stack, instead got {len(images.shape)} dimensions"

    if len(images.shape) == 3:
        k, m, n = images.shape
        z = None
        pixel_size = m * n
    else:
        k, z, m, n = images.shape
        pixel_size = z * m * n

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
            cdf = np.cumsum(cnt) / pixel_size

            interpolated = np.interp(cdf, cdfs[match_ix], values[match_ix])
            images[i, ...] = interpolated[ix]

        if i == 0 or match == "neighbor":
            val, cnt = np.unique(images[i, ...].flatten(), return_counts=True)
            cdf = np.cumsum(cnt) / pixel_size
            values.append(val)
            cdfs.append(cdf)

    if z is None:
        images = images.reshape(k, m, n)
    else:
        images = images.reshape(k, z, m, n)
    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)
