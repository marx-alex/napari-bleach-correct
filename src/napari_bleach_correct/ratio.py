from typing import Optional, Tuple

import numpy as np


def ratio_correct(
        images: np.ndarray,
        contrast_limits: Tuple[int, int],
        background_intens: Optional[float] = None
) -> np.ndarray:
    """
    Bleaching correction by applying a simple ratio method.

    .. math::
        I_i(x,y) = \frac{\bar{I}_0 - I_b}{\bar{I}_i - I_b}(I_i(x,y) - I_b)

    :param images: Image stack of shape `N, H, W`.
    :param contrast_limits: Lower and upper intensity bound.
    :param background_intens: Background intensity.
    """
    # cache image dtype
    dtype = images.dtype

    # scale image between 0 and 1
    images = images / contrast_limits[1]

    assert (
            len(images.shape) == 3
    ), f"Expected 3d image stack, instead got {len(images.shape)} dimensions"

    if background_intens is not None:
        assert (
                0 <= background_intens <= 1
        ), f"`background_intens` expected to be between 0 and 1, instead got {background_intens}"

    # calculate the mean intensity for every frame
    # store the intensity from the first frame
    I_mean = np.mean(images, axis=(1, 2))
    I_null = I_mean[0]

    # get the ratio for every frame
    if background_intens is None:
        background_intens = 0
    I_ratio = (I_null - background_intens) / (I_mean - background_intens)

    # subtract background from every pixel
    images = images - background_intens

    # multiply every frame by its ratio
    images = I_ratio.reshape(-1, 1, 1) * images

    # rescale and avoid overflow
    images = images * contrast_limits[1]
    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)
