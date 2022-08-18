"""
The gpustat module.
"""

__version__ = '1.0.0rc1'

from .core import GPUStat, GPUStatCollection
from .core import new_query
from .cli import print_gpustat, main, get_parser, nonnegative_int


__all__ = (
    '__version__',
    'GPUStat', 'GPUStatCollection',
    'new_query',
    'print_gpustat', 'main',
)
