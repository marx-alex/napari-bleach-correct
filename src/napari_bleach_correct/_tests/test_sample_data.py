import numpy as np
from napari_bleach_correct import make_sample_data


def test_sample_data():
    images = make_sample_data()

    assert isinstance(images, list)
    assert isinstance(images[0][0], np.ndarray)
