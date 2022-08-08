"""Imports pynvml with sanity checks and custom patches."""

import textwrap
import os


pynvml = None

# If this environment variable is set, we will bypass pynvml version validation
# so that legacy pynvml (nvidia-ml-py3) can be used. This would be useful
# in a case where there are conflicts on pynvml dependencies.
# However, beware that pynvml might produce wrong results (see #107).
ALLOW_LEGACY_PYNVML = os.getenv("ALLOW_LEGACY_PYNVML", "")
ALLOW_LEGACY_PYNVML = ALLOW_LEGACY_PYNVML.lower() not in ('false', '0', '')


try:
    # Check pynvml version: we require 11.450.129 or newer.
    # https://github.com/wookayin/gpustat/pull/107
    import pynvml
    if not (
        # Requires nvidia-ml-py >= 11.460.79
        hasattr(pynvml, 'NVML_BRAND_NVIDIA_RTX') or
        # Requires nvidia-ml-py >= 11.450.129, < 11.510.69
        hasattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses_v2')
    ) and not ALLOW_LEGACY_PYNVML:
        raise RuntimeError("pynvml library is outdated.")
except (ImportError, SyntaxError, RuntimeError) as e:
    raise ImportError(textwrap.dedent(
        """\
        pynvml is missing or an outdated version is installed.

        We require nvidia-ml-py>=11.450.129, and nvidia-ml-py3 shall not be used.
        For more details, please refer to: https://github.com/wookayin/gpustat/issues/107

        Your pynvml installation: """ + repr(pynvml) +
        """

        -----------------------------------------------------------
        Please reinstall `gpustat`:

        $ pip install --force-reinstall gpustat

        If it still does not fix the problem, please manually fix nvidia-ml-py installation:

        $ pip uninstall nvidia-ml-py3
        $ pip install --force-reinstall 'nvidia-ml-py<=11.495.46'
        """)) from e


__all__ = ['pynvml']
