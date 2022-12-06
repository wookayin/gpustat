"""
The gpustat module.
"""

# isort: skip_file
try:
    from ._version import version as __version__
    from ._version import version_tuple as __version_tuple__
except (ImportError, AttributeError) as ex:
    raise ImportError(
        "Unable to find `gpustat.__version__` string. "
        "Please try reinstalling gpustat; or if you are on a development "
        "version, then run `pip install -e .` and try again."
    ) from ex

from .core import GPUStat, GPUStatCollection
from .core import new_query, gpu_count, is_available
from .cli import print_gpustat, main


__all__ = (
    '__version__',
    'GPUStat',
    'GPUStatCollection',
    'new_query',
    'gpu_count',
    'is_available',
    'print_gpustat',
    'main',
)
