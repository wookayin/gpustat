"""
The gpustat module.
"""

__version__ = '1.0.0.dev0'

from .core import GPUStat, GPUStatCollection
from .core import new_query
from .cli import print_gpustat, main


__all__ = (
    '__version__',
    'GPUStat', 'GPUStatCollection',
    'new_query',
    'print_gpustat', 'main',
)
