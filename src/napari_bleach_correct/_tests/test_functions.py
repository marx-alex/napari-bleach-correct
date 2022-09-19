from ..modules import ratio_correct, exponential_correct, histogram_correct
import numpy as np


def test_ratio():
    images = np.random.random((5, 100, 100))
    corrected = ratio_correct(images, contrast_limits=(0, 1))

    assert isinstance(corrected, np.ndarray)


def test_exponential():
    images = np.random.random((5, 100, 100))
    corrected = exponential_correct(images, contrast_limits=(0, 1))

    assert isinstance(corrected, np.ndarray)


def test_histogram():
    images = np.random.random((5, 100, 100))
    corrected = histogram_correct(images, contrast_limits=(0, 1))

    assert isinstance(corrected, np.ndarray)
