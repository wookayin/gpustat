"""Imports pynvml with sanity checks and custom patches."""

import textwrap

try:
    import pynvml
    if not hasattr(pynvml, 'NVML_BRAND_TITAN_RTX'):
        raise RuntimeError("pynvml library is outdated.")
except (ImportError, SyntaxError, RuntimeError) as e:
    raise ImportError(textwrap.dedent(
        """\
        pynvml is missing or an outdated version is installed.

        Please reinstall `gpustat`:
        $ pip install -I gpustat

        or manually fix the package:
        $ pip uninstall nvidia-ml-py3
        $ pip install -I 'nvidia-ml-py!=375.*'
        """)) from e


__all__ = ['pynvml']
