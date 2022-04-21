from typing import Optional

from napari.layers import Image
from napari.types import LayerDataTuple
from magicgui import magicgui

from napari_bleach_correct.modules import ratio_correct, exponential_correct, histogram_correct


@magicgui(
    call_button="Correct",
    layer={
        "label": "Image Layer",
        "tooltip": "A 3d or 4d image layer in your Viewer"
    },
    background_intensity={
        "min": 0, "max": 1, "step": 0.05,
        "label": "Background Intensity",
        "tooltip": "Estimated background intensity as a normalized value between 0 and 1"
    }
)
def ratio_correct_widget(
        layer: Image,
        background_intensity: Optional[float] = None
) -> LayerDataTuple:
    """
    Bleaching correction by applying a simple ratio method.

    .. math::
        I_i(x,y) = \frac{\bar{I}_0 - I_b}{\bar{I}_i - I_b}(I_i(x,y) - I_b)

    Parameters
    ----------
    layer: napari.layers.Image
        3d image stack of shape `N, H, W` or
        4d image stack of shape `N, Z, H, W`.
    background_intensity: float
        Background intensity.

    Returns
    -------
    napari.types.LayerDataTuple
        New Image layer with the corrected images.
    """
    data = layer.data
    contrast_limits = layer.contrast_limits

    # correction name
    name = layer.name + " Corrected (Ratio Method)"

    # store metadata
    md = layer._metadata
    md.update({"method": "ratio", "background_intensity": background_intensity})

    corrected = ratio_correct(
        images=data,
        contrast_limits=contrast_limits,
        background_intens=background_intensity)

    return [(corrected, {"metadata": md, "name": name, "colormap": layer.colormap}, layer._type_string)]


@magicgui(
    call_button="Correct",
    layer={
        "label": "Image Layer",
        "tooltip": "A 3d or 4d image layer in your Viewer"
    },
    method={
        "choices": ["mono", "bi"],
        "label": "Exponential Curve",
        "tooltip": "The type of the exponential curve to use for fitting"
    }
)
def exponential_correct_widget(
        layer: Image,
        method: str = "bi"
) -> LayerDataTuple:
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

    Parameters
    ----------
    layer: napari.layers.Image
        3d image stack of shape `N, H, W` or
        4d image stack of shape `N, Z, H, W`.
    method: str
        Type of exponential curve ("mono" or "bi").

    Returns
    -------
    napari.types.LayerDataTuple
        New Image layer with the corrected images.
    """
    data = layer.data
    contrast_limits = layer.contrast_limits

    # correction name
    name = layer.name + " Corrected (Exponential Curve Method)"

    # store metadata
    md = layer._metadata
    md.update({"method": "exponential", "curve_type": method})

    corrected = exponential_correct(
        images=data,
        contrast_limits=contrast_limits,
        method=method)

    return [(corrected, {"metadata": md, "name": name, "colormap": layer.colormap}, layer._type_string)]


@magicgui(
    call_button="Correct",
    layer={
        "label": "Image Layer",
        "tooltip": "A 3d or 4d image layer in your Viewer"
    },
    match={
        "choices": ["first", "neighbor"],
        "label": "Reference Frame",
        "tooltip": "Match histogram to the first our the neighboring frame"
    }
)
def histogram_correct_widget(
        layer: Image,
        match: str = "neighbor"
) -> LayerDataTuple:
    """
    Bleaching correction by matching histograms to a reference image.

    The correct pixel values can be calculated by the cumulative distribution function
    of a frame and its reference frame.

    .. math::
        p' = CDF_{ref}^{-1}(CDF_i(p))

    Parameters
    ----------
    layer: napari.layers.Image
        3d image stack of shape `N, H, W` or
        4d image stack of shape `N, Z, H, W`.
    match: str
        Match frame histogram with 'first' our 'neighbor' histogram.

    Returns
    -------
    napari.types.LayerDataTuple
        New Image layer with the corrected images.

    References
    ----------
    Miura K. Bleach correction ImageJ plugin for compensating the photobleaching of
    time-lapse sequences. F1000Res. 2020 Dec 21;9:1494.
    doi: 10.12688/f1000research.27171.1
    """
    data = layer.data
    contrast_limits = layer.contrast_limits

    # correction name
    name = layer.name + " Corrected (Histogram Matching Method)"

    # store metadata
    md = layer._metadata
    md.update({"method": "histogram", "match": match})

    corrected = histogram_correct(
        images=data,
        contrast_limits=contrast_limits,
        match=match)

    return [(corrected, {"metadata": md, "name": name, "colormap": layer.colormap}, layer._type_string)]
