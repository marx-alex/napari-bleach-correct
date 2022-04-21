from collections import OrderedDict

from qtpy.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QComboBox, QLabel, QPushButton
from qtpy.QtCore import Qt
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import napari
from napari.layers import Image
import numpy as np


cmap = [
    "#3a86ff",
    "#ff006e",
    "#ffbe0b",
    "#8338ec",
    "#fb5607"
]


class IntensityPlotWidget(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self._viewer = viewer
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignBottom)

        # description
        widget = QLabel("Choose two images to compare:")
        font = widget.font()
        font.setPointSize(10)
        widget.setFont(font)
        widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(widget)

        # open image layers
        choices = [layer.name for layer in self._viewer.layers if isinstance(layer, Image)]

        # get image layers
        image_layout = QFormLayout()
        self.layer1, self.layer2 = QComboBox(), QComboBox()
        self.layer1.addItems(choices)
        self.layer2.addItems(choices)

        self._viewer.layers.events.inserted.connect(self._on_layer_added)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        image_layout.addRow("Image 1", self.layer1)
        image_layout.addRow("Image 2", self.layer2)

        layout.addLayout(image_layout)

        # push button
        button = QPushButton("Plot")
        button.clicked.connect(self._plot)
        layout.addWidget(button)

        self.setLayout(layout)
        self.setWindowTitle("Plot mean intensities")

    def _on_layer_added(self, event):
        if isinstance(event.value, Image):
            self.layer1.addItem(event.value.name)
            self.layer2.addItem(event.value.name)

    def _on_layer_removed(self, event):
        if isinstance(event.value, Image):
            index1 = self.layer1.findText(event.value.name)
            index2 = self.layer2.findText(event.value.name)
            self.layer1.removeItem(index1)
            self.layer2.removeItem(index2)

    def _plot(self):
        data = self._get_data()
        plot_widget = IntensityPlot(data=data, xaxis="xaxis", yaxis="yaxis")
        self._viewer.window.add_dock_widget(plot_widget, area="bottom", name="Intensity Plot")

    def _get_data(self):
        name1 = self.layer1.currentText()
        name2 = self.layer2.currentText()
        img1 = self._viewer.layers[name1].data
        img2 = self._viewer.layers[name2].data

        assert (
            len(img1.shape) == len(img2.shape)
        ), "Chosen layers are not of the same dimension"

        # calculate the mean intensity for every frame
        axes1 = tuple([i for i in range(len(img1.shape))])
        I_mean1 = np.mean(img1, axis=axes1[1:])
        x_data1 = np.arange(len(I_mean1))
        axes2 = tuple([i for i in range(len(img2.shape))])
        I_mean2 = np.mean(img2, axis=axes2[1:])
        x_data2 = np.arange(len(I_mean2))

        data = OrderedDict()
        data[name1] = dict(
                xaxis=x_data1,
                yaxis=I_mean1
            )
        data[name2] = dict(
                xaxis=x_data2,
                yaxis=I_mean2
            )
        return data


class IntensityPlot(PlotWidget):
    def __init__(self, data: OrderedDict, xaxis: str, yaxis: str):
        super().__init__()

        self.setLabel('left', 'Mean Intensity')
        self.setLabel('bottom', 'Frame')
        self.addLegend()

        for i, name in enumerate(data):
            if i > len(cmap):
                i = i % len(cmap)
            pen = pg.mkPen(color=cmap[i])
            self.plot(data[name][xaxis], data[name][yaxis], pen=pen, name=name)
