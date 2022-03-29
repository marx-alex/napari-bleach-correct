
__version__ = "0.0.1"

from .ratio import ratio_correct
from .exponential import exponential_correct
from .histogram import histogram_correct

from ._sample_data import make_sample_data 
from ._widget import ratio_correct_widget, exponential_correct_widget, histogram_correct_widget
