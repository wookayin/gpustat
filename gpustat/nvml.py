"""Imports pynvml with sanity checks and custom patches."""

import textwrap


pynvml = None

try:
    # Check pynvml version: we require 11.450.129 or newer.
    # https://github.com/wookayin/gpustat/pull/107
    import pynvml
    if not (
        # Requires nvidia-ml-py >= 11.460.79
        hasattr(pynvml, 'NVML_BRAND_NVIDIA_RTX') or
        # Requires nvidia-ml-py >= 11.450.129, < 11.510.69
        hasattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses_v2')
    ):
        raise RuntimeError("pynvml library is outdated.")
except (ImportError, SyntaxError, RuntimeError) as e:
    raise ImportError(textwrap.dedent(
        """\
        pynvml is missing or an outdated version is installed.

        We require nvidia-ml-py>=11.450.129; see GH-107 for more details.
        Your pynvml installation: """ + repr(pynvml) +
        """

        -----------------------------------------------------------
        Please reinstall `gpustat`:

        $ pip install --force-reinstall gpustat

        if it still does not fix the problem, manually fix nvidia-ml-py installation:

        $ pip uninstall nvidia-ml-py3
        $ pip install --force-reinstall 'nvidia-ml-py<=11.495.46'
        """)) from e


__all__ = ['pynvml']
