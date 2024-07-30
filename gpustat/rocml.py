"""Imports pyrsmi and wraps it in a pynvml compatible interface."""

# pylint: disable=protected-access

import atexit
import functools
import os
import sys
import textwrap
import warnings

from collections import namedtuple


from pyrsmi import rocml

NVML_TEMPERATURE_GPU = 1

class NVMLError_Unknown(Exception):
    def __init__(self, message="An unknown ROCMLError has occurred"):
        self.message = message
        super().__init__(self.message)

class NVMLError_GpuIsLost(Exception):
    def __init__(self, message="ROCM Device is lost."):
        self.message = message
        super().__init__(self.message)


_stdout_dup = os.dup(1)
_stderr_dup = os.dup(2)
_silent_pipe = os.open(os.devnull, os.O_WRONLY)

def silent_run(to_call, *args, **kwargs):
    os.dup2(_silent_pipe, 1)
    os.dup2(_silent_pipe, 2)
    retval = to_call(*args, **kwargs)
    os.dup2(_stdout_dup, 1)
    os.dup2(_stderr_dup, 2)
    return retval

def nvmlDeviceGetCount():
    return silent_run(rocml.smi_get_device_count)

def nvmlDeviceGetHandleByIndex(dev):
    return dev

def nvmlDeviceGetIndex(dev):
    return dev

def nvmlDeviceGetName(dev):
    return silent_run(rocml.smi_get_device_name, dev)

def nvmlDeviceGetUUID(dev):
    return silent_run(rocml.smi_get_device_uuid, dev)

def nvmlDeviceGetTemperature(dev, loc=NVML_TEMPERATURE_GPU):
    return silent_run(rocml.smi_get_device_temp, dev, loc)

def nvmlSystemGetDriverVersion():
    return silent_run(rocml.smi_get_kernel_version)

def check_driver_nvml_version(driver_version_str: str):
    """Show warnings when an incompatible driver is used."""

    def safeint(v) -> int:
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    driver_version = tuple(safeint(v) for v in
                           driver_version_str.strip().split("."))

    if driver_version < (6, 7, 8):
        warnings.warn(f"This version of ROCM Driver {driver_version_str} is untested, ")

def nvmlDeviceGetFanSpeed(dev):
    return silent_run(rocml.smi_get_device_fan_speed, dev)

MemoryInfo = namedtuple('MemoryInfo', ['total', 'used'])

def nvmlDeviceGetMemoryInfo(dev):
    return MemoryInfo(total=silent_run(rocml.smi_get_device_memory_total, dev),
                      used=silent_run(rocml.smi_get_device_memory_used, dev))

UtilizationRates = namedtuple('UtilizationRates', ['gpu'])

def nvmlDeviceGetUtilizationRates(dev):
    return UtilizationRates(gpu=silent_run(rocml.smi_get_device_utilization, dev))

def nvmlDeviceGetEncoderUtilization(dev):
    return None

def nvmlDeviceGetDecoderUtilization(dev):
    return None

def nvmlDeviceGetPowerUsage(dev):
    return silent_run(rocml.smi_get_device_average_power, dev)

def nvmlDeviceGetEnforcedPowerLimit(dev):
    return None

ComputeProcess = namedtuple('ComputeProcess', ['pid'])

def nvmlDeviceGetComputeRunningProcesses(dev):
    processes = silent_run(rocml.smi_get_device_compute_process)
    return [ComputeProcess(pid=i) for i in processes]

def nvmlDeviceGetGraphicsRunningProcesses(dev):
    return None

# Upon importing this module, let rocml be initialized and remain active
# throughout the lifespan of the python process (until gpustat exists).
_initialized: bool
_init_error = None
try:
    rocml.smi_initialize()
    _initialized = True

    def _shutdown():
        rocml.smi_shutdown()
    atexit.register(_shutdown)

except Exception as exc:
    _initialized = False
    _init_error = exc


def ensure_initialized():
    if not _initialized:
        raise _init_error  # type: ignore



