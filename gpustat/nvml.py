"""Imports pynvml with sanity checks and custom patches."""

# pylint: disable=protected-access

import atexit
import functools
import os
import sys
import textwrap
import warnings

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
        raise ImportError("pynvml library is outdated.")

    if not hasattr(pynvml, '_nvmlGetFunctionPointer'):
        raise ImportError(
            """pynvml appears to be a non-official package.

            The PyPI package `pynvml` should not be used.
            Please use the official NVIDIA bindings `nvidia-ml-py`.
            """)

except (ImportError, SyntaxError, RuntimeError) as e:
    _pynvml = sys.modules.get('pynvml', None)

    raise ImportError(textwrap.dedent(
        """\
        pynvml is missing or an outdated version is installed.

        gpustat requires nvidia-ml-py>=11.450.129, and the *official* NVIDIA python bindings
        should be used; neither nvidia-ml-py3 nor gpuopenanalytics/pynvml is compatible.

        For more details, please refer to:
            https://github.com/wookayin/gpustat/issues/107
            https://github.com/wookayin/gpustat/issues/153

        The root cause: """ + str(e) +
        """

        Your pynvml installation: """ + repr(_pynvml) +
        """

        -----------------------------------------------------------
        (Suggested Fix) Please reinstall `gpustat`:

        $ pip install --force-reinstall gpustat

        If it still does not fix the problem, please uninstall `pynvml` packages and reinstall `nvidia-ml-py` manually:

        $ pip uninstall nvidia-ml-py3 pynvml
        $ pip install --force-reinstall --ignore-installed 'nvidia-ml-py'
        """)) from e


class NvidiaCompatibilityWarning(UserWarning):
    pass


def check_driver_nvml_version(driver_version_str: str):
    """Show warnings when an incompatible driver is used."""

    def safeint(v) -> int:
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    driver_version = tuple(safeint(v) for v in
                           driver_version_str.strip().split("."))

    # #161: invalid process information on 535.xx
    is_pynvml_535_77 = (
        hasattr(pynvml.c_nvmlProcessInfo_t, 'usedGpuCcProtectedMemory') and
        # Note: __name__ changed to pynvml.c_nvmlProcessInfo_v2_t since 12.535.108+
        pynvml.c_nvmlProcessInfo_t.__name__ == 'c_nvmlProcessInfo_t'
    )

    if (535, 43) <= driver_version < (535, 86):
        # See #161: these are buggy, gives wrong process information
        # except for nvidia-ml-py == 12.535.77 (which is a buggy version too).
        # Note: NVIDIA 535.86+ and nvidia-ml-py 12.535.108+ fixes the bug
        if not is_pynvml_535_77:
            warnings.warn(
                f"This version of NVIDIA Driver {driver_version_str} is incompatible, "
                "process information will be inaccurate. "
                "Upgrade the NVIDIA driver to 535.104.05 or higher, "
                "or use nvidia-ml-py==12.535.77. For more details, see "
                "https://github.com/wookayin/gpustat/issues/161.",
                category=NvidiaCompatibilityWarning, stacklevel=2)
    else:
        if is_pynvml_535_77:   # pynvml 12.535.77 should not be used
            warnings.warn(
                "This version of nvidia-ml-py (possibly 12.535.77) is incompatible. "
                "Please upgrade nvidia-ml-py to the latest version. "
                "(pip install --upgrade --force-reinstall nvidia-ml-py)",
                category=NvidiaCompatibilityWarning, stacklevel=2)


# Monkey-patch nvml due to breaking changes in pynvml.
# See #107, #141, and test_gpustat.py for more details.

_original_nvmlGetFunctionPointer = pynvml._nvmlGetFunctionPointer
_original_nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo


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
            test_gpustat::NvidiaDriverMock for test scenarios. See #107.
        """
        M = pynvml_monkeypatch

        try:
            ret = M.original_nvmlGetFunctionPointer(name)
            return ret
        except pynvml.NVMLError_FunctionNotFound:  # type: ignore
            if name in M.FUNCTION_FALLBACKS:
                # Lack of ...Processes_v3 APIs happens for
                # OLD drivers < 510.39.01 && pynvml >= 11.510, where
                # we fallback to v2 APIs. (see #107 for more details)

                ret = M.original_nvmlGetFunctionPointer(
                    M.FUNCTION_FALLBACKS[name]
                )
                # populate the cache, so this handler won't get executed again
                pynvml._nvmlGetFunctionPointer_cache[name] = ret

            else:
                # Unknown case, cannot handle. re-raise again
                raise

        return ret

    @staticmethod  # Note: must be defined as a staticmethod to allow mocking.
    def original_nvmlDeviceGetMemoryInfo(*args, **kwargs):
        return _original_nvmlDeviceGetMemoryInfo(*args, **kwargs)

    has_memoryinfo_v2 = None

    @staticmethod
    @functools.wraps(pynvml.nvmlDeviceGetMemoryInfo)
    def nvmlDeviceGetMemoryInfo(handle):
        """A patched version of nvmlDeviceGetMemoryInfo.

        This tries `version=N.nvmlMemory_v2` if the nvmlDeviceGetMemoryInfo_v2
        function is available (for driver >= 515), or fallback to the legacy
        v1 API for (driver < 515) to yield a correct result. See #141.
        """
        M = pynvml_monkeypatch

        if M.has_memoryinfo_v2 is None:
            try:
                pynvml._nvmlGetFunctionPointer("nvmlDeviceGetMemoryInfo_v2")
                M.has_memoryinfo_v2 = True
            except pynvml.NVMLError_FunctionNotFound:  # type: ignore
                M.has_memoryinfo_v2 = False

        if hasattr(pynvml, 'nvmlMemory_v2'):  # pynvml >= 11.510.69
            try:
                memory = M.original_nvmlDeviceGetMemoryInfo(
                    handle, version=pynvml.nvmlMemory_v2)
            except pynvml.NVMLError_FunctionNotFound:  # type: ignore
                # pynvml >= 11.510 but driver is old (<515.39)
                memory = M.original_nvmlDeviceGetMemoryInfo(handle)
        else:
            if M.has_memoryinfo_v2:
                warnings.warn(
                    "Your NVIDIA driver requires a compatible version of "
                    "pynvml (>= 11.510.69) installed to display the correct "
                    "memory usage information (See #141 for more details). "
                    "Please try `pip install --upgrade nvidia-ml-py`.",
                    category=UserWarning)
            memory = M.original_nvmlDeviceGetMemoryInfo(handle)

        return memory


setattr(pynvml, '_nvmlGetFunctionPointer', pynvml_monkeypatch._nvmlGetFunctionPointer)
setattr(pynvml, 'nvmlDeviceGetMemoryInfo', pynvml_monkeypatch.nvmlDeviceGetMemoryInfo)


# Upon importing this module, let pynvml be initialized and remain active
# throughout the lifespan of the python process (until gpustat exists).
_initialized: bool
_init_error = None
try:
    pynvml.nvmlInit()
    _initialized = True

    def _shutdown():
        pynvml.nvmlShutdown()
    atexit.register(_shutdown)

except pynvml.NVMLError as exc:
    _initialized = False
    _init_error = exc


def ensure_initialized():
    if not _initialized:
        raise _init_error  # type: ignore


__all__ = [
    'pynvml',
    'check_driver_nvml_version',
    'ensure_initialized',
]
