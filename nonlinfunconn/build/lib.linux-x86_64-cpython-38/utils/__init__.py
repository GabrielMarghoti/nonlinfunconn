from ._integration import integral
from .integration import integral as integral_py
from ._convolution import convolution1, convolution, slice_test
from .irrarray import irrarray

__all__ = ['irrarray', 'convolution', 'integration']