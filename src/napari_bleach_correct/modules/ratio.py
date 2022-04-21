from typing import Optional, Tuple

import numpy as np
from napari.types import ImageData


def ratio_correct(
        images: ImageData,
        contrast_limits: Tuple[int, int],
        background_intens: Optional[float] = None
) -> ImageData:
    # cache image dtype
    dtype = images.dtype

    # scale image between 0 and 1
    images = images / contrast_limits[1]

    assert (
            3 <= len(images.shape) <= 4
    ), f"Expected 3d or 4d image stack, instead got {len(images.shape)} dimensions"

    if background_intens is not None:
        assert (
                0 <= background_intens <= 1
        ), f"`background_intens` expected to be between 0 and 1, instead got {background_intens}"

    # calculate the mean intensity for every frame
    # store the intensity from the first frame
    axes = tuple([i for i in range(len(images.shape))])
    I_mean = np.mean(images, axis=axes[1:])
    I_null = I_mean[0]

    # get the ratio for every frame
    if background_intens is None:
        background_intens = 0
    I_ratio = (I_null - background_intens) / (I_mean - background_intens)

    # subtract background from every pixel
    images = images - background_intens

    # multiply every frame by its ratio
    if len(images.shape) == 3:
        I_ratio = I_ratio.reshape(-1, 1, 1)
    else:
        I_ratio = I_ratio.reshape(-1, 1, 1, 1)
    images = I_ratio * images

    # rescale and avoid overflow
    images = images * contrast_limits[1]
    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)
