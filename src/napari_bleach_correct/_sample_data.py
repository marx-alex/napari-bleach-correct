from pathlib import Path

from skimage import io
import numpy as np


def make_sample_data():
    """
    Open sample bleaching data.

    This opens low resolution images of mitochondria in living cardiomyocytes.
    The images have been taken over 47 hours in a 1-hour cycle.
    """
    path = Path.cwd()

    image_paths = sorted(
        list(
            path.rglob("./data/mito*.png")
        ), key=lambda x: int(x.stem.split("_")[1])
    )

    imgs = [io.imread(image_path, as_gray=True) for image_path in image_paths]
    imgs = np.stack(imgs)
    return [(imgs, {"name": "Bleaching Mito", "colormap": "red"})]
