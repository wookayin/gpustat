"""
The gpustat module.
"""

__version__ = '0.5.0.dev1'

from .core import GPUStat, GPUStatCollection
from .core import new_query
from .__main__ import print_gpustat, main


__all__ = (
    'GPUStat', 'GPUStatCollection',
    'new_query',
    'print_gpustat', 'main',
)
