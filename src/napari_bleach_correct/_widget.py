from typing import Optional

from napari.layers import Image
from napari.types import LayerDataTuple
from magicgui import magic_factory

from napari_bleach_correct import ratio_correct, exponential_correct, histogram_correct

method_dict = {
    "Ratio": ratio_correct,
    "Exponential fitting": None,
    "Histogram matching": None
}


@magic_factory(call_button="Correct")
def ratio_correct_widget(
        layer: Image,
        background_intensity: Optional[float] = None
) -> LayerDataTuple:
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


@magic_factory(call_button="Correct", method={"choices": ["mono", "bi"]})
def exponential_correct_widget(
        layer: Image,
        method: str = "bi"
) -> LayerDataTuple:
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


@magic_factory(call_button="Correct", match={"choices": ["first", "neighbor"]})
def histogram_correct_widget(
        layer: Image,
        match: str = "neighbor"
) -> LayerDataTuple:
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
