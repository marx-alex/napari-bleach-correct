from typing import Callable
from dataclasses import dataclass

import napari
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from qtpy.QtCore import Qt

from ._button_grid import ButtonGrid
from ._widgets import ratio_correct_widget, exponential_correct_widget, histogram_correct_widget
from ._plot_widget import IntensityPlotWidget


@dataclass
class Category:
    widget: Callable
    tool_tip: str = ""


CATEGORIES = {
    "Ratio Method": Category(
        widget=ratio_correct_widget,
        tool_tip="Applying a simple intensity ratio",
    ),
    "Exponential Curve Fitting": Category(
        widget=exponential_correct_widget,
        tool_tip="Fitting a mono- or bi-exponential curve",
    ),
    "Histogram matching": Category(
        widget=histogram_correct_widget,
        tool_tip="Matching histograms to the first or neighboring frame"
    ),
    "Plot Mean Intensities": Category(
        widget=IntensityPlotWidget,
        tool_tip="Plot Mean Intensities of two image layers"
    )
}


class MainWidget(QWidget):
    """
    Main Widget for napari-bleach-correct.
    """
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self._viewer = viewer
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        widget = QLabel("Choose a bleaching correction method:")
        font = widget.font()
        font.setPointSize(10)
        widget.setFont(font)
        widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(widget)

        icon_grid = ButtonGrid(self)
        icon_grid.addItems(CATEGORIES)
        icon_grid.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(icon_grid)

        self.setLayout(layout)
        self.setWindowTitle("Bleaching Correction")

    def _on_item_clicked(self, item):
        name = item.text()
        widget = CATEGORIES[name].widget

        if name == "Plot Mean Intensities":
            widget = widget(self._viewer)

        self._viewer.window.add_dock_widget(widget, area="right", name=name)
