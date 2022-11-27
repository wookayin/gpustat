"""Imports pynvml with sanity checks and custom patches."""

import functools
import os
import sys
import textwrap

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
    _pynvml = sys.modules.get('pynvml', None)

    raise ImportError(textwrap.dedent(
        """\
        pynvml is missing or an outdated version is installed.

        We require nvidia-ml-py>=11.450.129, and nvidia-ml-py3 shall not be used.
        For more details, please refer to: https://github.com/wookayin/gpustat/issues/107

        Your pynvml installation: """ + repr(_pynvml) +
        """

        -----------------------------------------------------------
        Please reinstall `gpustat`:

        $ pip install --force-reinstall gpustat

        If it still does not fix the problem, please manually fix nvidia-ml-py installation:

        $ pip uninstall nvidia-ml-py3
        $ pip install --force-reinstall 'nvidia-ml-py<=11.495.46'
        """)) from e


# Monkey-patch nvml due to breaking changes in pynvml.
# See #107,  #141, and test_gpustat.py for more details.

_original_nvmlGetFunctionPointer = pynvml._nvmlGetFunctionPointer


class pynvml_monkeypatch:

    @staticmethod  # Note: must be defined as a staticmethod to allow mocking.
    def original_nvmlGetFunctionPointer(name):
        return _original_nvmlGetFunctionPointer(name)

    FUNCTION_FALLBACKS = {
        # for pynvml._nvmlGetFunctionPointer
        'nvmlDeviceGetComputeRunningProcesses_v3': 'nvmlDeviceGetComputeRunningProcesses_v2',
        'nvmlDeviceGetGraphicsRunningProcesses_v3': 'nvmlDeviceGetGraphicsRunningProcesses_v2',
    }

    @staticmethod
    @functools.wraps(pynvml._nvmlGetFunctionPointer)
    def _nvmlGetFunctionPointer(name):
        """Our monkey-patched pynvml._nvmlGetFunctionPointer().

        See also:
            test_gpustat::NvidiaDriverMock for test scenarios
        """

        try:
            ret = pynvml_monkeypatch.original_nvmlGetFunctionPointer(name)
            return ret
        except pynvml.NVMLError as e:
            if e.value != pynvml.NVML_ERROR_FUNCTION_NOT_FOUND:  # type: ignore
                raise

            if name in pynvml_monkeypatch.FUNCTION_FALLBACKS:
                # Lack of ...Processes_v3 APIs happens for
                # OLD drivers < 510.39.01 && pynvml >= 11.510, where
                # we fallback to v2 APIs. (see #107 for more details)

                ret = pynvml_monkeypatch.original_nvmlGetFunctionPointer(
                    pynvml_monkeypatch.FUNCTION_FALLBACKS[name]
                )
                # populate the cache, so this handler won't get executed again
                pynvml._nvmlGetFunctionPointer_cache[name] = ret

            else:
                # Unknown case, cannot handle. re-raise again
                raise

        return ret


setattr(pynvml, '_nvmlGetFunctionPointer',
        pynvml_monkeypatch._nvmlGetFunctionPointer)


__all__ = ['pynvml']
