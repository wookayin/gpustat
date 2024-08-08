"""Imports rocmi and wraps it in a pynvml compatible interface."""

# pylint: disable=protected-access

import atexit
import functools
import os
import sys
import textwrap
import warnings

from collections import namedtuple

try:
    # Check for rocmi.
    import rocmi
except (ImportError, SyntaxError, RuntimeError) as e:
    _rocmi = sys.modules.get("rocmi", None)

    raise ImportError(
        textwrap.dedent(
            """\
        rocmi is missing or an outdated version is installed.

        The root cause: """
            + str(e)
            + """

        Your rocmi installation: """
            + repr(_rocmi)
            + """

        -----------------------------------------------------------
        (Suggested Fix) Please install rocmi using pip.
        """
        )
    ) from e

NVML_TEMPERATURE_GPU = 1


class NVMLError(Exception):
    def __init__(self, message="ROCM Error"):
        self.message = message
        super().__init__(self.message)


class NVMLError_Unknown(Exception):
    def __init__(self, message="An unknown ROCM Error has occurred"):
        self.message = message
        super().__init__(self.message)


class NVMLError_GpuIsLost(Exception):
    def __init__(self, message="ROCM Device is lost."):
        self.message = message
        super().__init__(self.message)


def nvmlDeviceGetCount():
    return len(rocmi.get_devices())


def nvmlDeviceGetHandleByIndex(dev):
    return rocmi.get_devices()[dev]


def nvmlDeviceGetIndex(handle):
    for i, d in enumerate(rocmi.get_devices()):
        if d.bus_id == handle.bus_id:
            return i

    return -1


def nvmlDeviceGetName(handle):
    return handle.name


def nvmlDeviceGetUUID(handle):
    return handle.unique_id


def nvmlDeviceGetTemperature(handle, loc=NVML_TEMPERATURE_GPU):
    metrics = handle.get_metrics()
    return metrics.temperature_hotspot


def nvmlSystemGetDriverVersion():
    return ""


def check_driver_nvml_version(driver_version_str: str):
    """Show warnings when an incompatible driver is used."""

    def safeint(v) -> int:
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    driver_version = tuple(safeint(v) for v in driver_version_str.strip().split("."))

    if len(driver_version) == 0 or driver_version <= (0,):
        return
    if driver_version < (6, 7, 8):
        warnings.warn(f"This version of ROCM Driver {driver_version_str} is untested, ")


def nvmlDeviceGetFanSpeed(handle):
    try:
        speed = handle.get_metrics().current_fan_speed
    except AttributeError:
        return None

    return speed


MemoryInfo = namedtuple("MemoryInfo", ["total", "used"])


def nvmlDeviceGetMemoryInfo(handle):

    return MemoryInfo(
        total=handle.vram_total,
        used=handle.vram_used,
    )


UtilizationRates = namedtuple("UtilizationRates", ["gpu"])


def nvmlDeviceGetUtilizationRates(handle):
    metrics = handle.get_metrics()
    return UtilizationRates(gpu=metrics.average_gfx_activity)


def nvmlDeviceGetEncoderUtilization(dev):
    return None


def nvmlDeviceGetDecoderUtilization(dev):
    return None


def nvmlDeviceGetPowerUsage(handle):
    return handle.current_power / 1000000


def nvmlDeviceGetEnforcedPowerLimit(handle):
    return handle.power_limit / 1000000


ComputeProcess = namedtuple("ComputeProcess", ["pid", "usedGpuMemory"])


def nvmlDeviceGetComputeRunningProcesses(handle):
    results = handle.get_processes()
    return [ComputeProcess(pid=x.pid, usedGpuMemory=x.vram_usage) for x in results]


def nvmlDeviceGetGraphicsRunningProcesses(dev):
    return None


def nvmlDeviceGetClockInfo(handle):
    metrics = handle.get_metrics()

    try:
        clk = metrics.current_gfxclks[0]
    except AttributeError:
        clk = metrics.current_gfxclk

    return clk


def nvmlDeviceGetMaxClockInfo(handle):
    return handle.get_clock_info()[-1]


# Upon importing this module, let rocmi be initialized and remain active
# throughout the lifespan of the python process (until gpustat exists).
_initialized: bool
_init_error = None
try:
    # rocmi_init() No init required.
    _initialized = True

    def _shutdown():
        # rocmi_shut_down() No shutdown required.
        pass

    atexit.register(_shutdown)

except Exception as exc:
    _initialized = False
    _init_error = exc


def ensure_initialized():
    if not _initialized:
        raise _init_error  # type: ignore
