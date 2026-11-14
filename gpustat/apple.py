"""Apple Silicon GPU backend (experimental).

Data sources, none of which require root privileges:

* ``ioreg``: the built-in GPU is an ``IOAccelerator`` entry whose class name
  starts with ``AGX``. It provides the model name and the driver-reported
  ``PerformanceStatistics`` (``Device Utilization %`` and
  ``In use system memory``). Its ``AGXDeviceUserClient`` children, one per
  GPU client connection, name the creating process and when it last
  submitted GPU work; recent submitters are listed as GPU processes.
* SMC ``Tg*`` keys: GPU temperature. The SMC interface is undocumented, so
  any failure only makes the temperature unsupported.

Apple Silicon uses unified memory, so ``memory.total`` is the host RAM size
and ``memory.used`` is the system memory the GPU driver reports as in use.
Power and per-process GPU memory are reported as unsupported.
"""

import ctypes
import platform
import plistlib
import struct
import subprocess
import sys
import threading
import time
from collections import namedtuple

import psutil

NVML_TEMPERATURE_GPU = 1

# The GPU shares host memory; see GPUStat.memory_unified.
UNIFIED_MEMORY = True

# -l -d 2 includes the GPU's AGXDeviceUserClient children: one call for all.
IOREG_COMMAND = ["ioreg", "-a", "-r", "-l", "-d", "2", "-c", "IOAccelerator"]

# A process counts as a GPU process if it submitted GPU work this recently.
# shortcut: a single GPU command longer than this hides its process (macOS
# exposes no in-flight state), upgrade if users report busy GPUs with no owner.
ACTIVE_PROCESS_WINDOW_NS = 2 * 10 ** 9

# Powered-down sensors can briefly report a few degrees; treat as invalid.
TEMPERATURE_RANGE = (10.0, 150.0)

_IOKIT_PATH = "/System/Library/Frameworks/IOKit.framework/IOKit"


class NVMLError(Exception):
    """Base error for the Apple GPU backend."""


class NVMLError_Unknown(NVMLError):
    """An unknown Apple GPU error occurred."""


class NVMLError_GpuIsLost(NVMLError):
    """The Apple GPU is unavailable."""


class NVMLError_NotSupported(NVMLError):
    """The requested metric is not supported."""


MemoryInfo = namedtuple("MemoryInfo", ["total", "used"])
UtilizationRates = namedtuple("UtilizationRates", ["gpu"])
# No usedGpuMemory: macOS does not attribute GPU memory to processes.
ComputeProcess = namedtuple("ComputeProcess", ["pid", "command"])

_state = threading.local()
_smc = None


def is_available() -> bool:
    return sys.platform == "darwin"


def active_processes(clients, now_ns):
    """Processes that submitted GPU work within the active window.

    ``IOUserClientCreator`` looks like ``"pid 404, WindowServer"`` (the name
    is truncated by the kernel). Each ``AppUsage`` item describes one command
    queue; ``lastSubmittedTime`` is in nanoseconds of the uptime clock.
    """
    active = {}
    for client in clients:
        pid, _, name = client.get("IOUserClientCreator", "")[4:].partition(",")
        if pid.isdigit() and any(
                0 <= now_ns - usage.get("lastSubmittedTime", 0)
                <= ACTIVE_PROCESS_WINDOW_NS
                for usage in client.get("AppUsage", [])):
            active[int(pid)] = name.strip()
    return [ComputeProcess(pid, name) for pid, name in sorted(active.items())]


def query():
    """Query the Apple GPU using one ``ioreg`` invocation."""
    if not is_available():
        raise NVMLError("the Apple backend needs macOS")
    try:
        output = subprocess.run(
            IOREG_COMMAND, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=10).stdout
        # ioreg prints nothing when no entry matches.
        entries = plistlib.loads(output) if output.strip() else []
        now_ns = time.clock_gettime_ns(8)  # CLOCK_UPTIME_RAW
    except Exception as exc:
        raise NVMLError("failed to query ioreg: {}".format(exc)) from exc

    devices = []
    for entry in entries:
        # Skip non-Apple accelerators (e.g. Intel or AMD GPUs on Intel Macs).
        if not entry.get("IOClass", "").startswith("AGX"):
            continue
        stats = entry.get("PerformanceStatistics", {})
        devices.append({
            "index": len(devices),
            "name": entry.get("model", "Apple GPU"),
            "uuid": "APPLE-GPU-{}".format(entry.get("IORegistryEntryID")),
            "utilization": stats.get("Device Utilization %"),
            "memory.used": stats.get("In use system memory"),
            "processes": active_processes(
                entry.get("IORegistryEntryChildren", []), now_ns),
        })
    if not devices:
        raise NVMLError("ioreg returned no Apple GPU")
    return devices


class _SMCKeyData(ctypes.Structure):
    """AppleSMC ``SMCKeyData_t`` (80 bytes); unused fields are padding."""

    _fields_ = [
        ("key", ctypes.c_uint32),
        ("_version_and_power_limit", ctypes.c_uint8 * 24),
        ("data_size", ctypes.c_uint32),
        ("data_type", ctypes.c_uint32),
        ("_data_attributes", ctypes.c_uint8 * 4),
        ("result", ctypes.c_uint8),
        ("_status", ctypes.c_uint8),
        ("command", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("bytes", ctypes.c_uint8 * 32),
    ]


_SMC_READ_KEY, _SMC_GET_KEY_FROM_INDEX, _SMC_GET_KEY_INFO = 5, 8, 9


def _fourcc(name):
    return struct.unpack(">I", name.encode("ascii"))[0]


class SMC:
    """Minimal read-only client for the AppleSMC user client."""

    def __init__(self):
        iokit = ctypes.CDLL(_IOKIT_PATH)
        u32 = ctypes.c_uint32
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceGetMatchingService.restype = u32
        iokit.IOServiceGetMatchingService.argtypes = [u32, ctypes.c_void_p]
        iokit.IOServiceOpen.argtypes = [u32, u32, u32, ctypes.POINTER(u32)]
        iokit.IOConnectCallStructMethod.argtypes = [
            u32, u32, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self._call_method = iokit.IOConnectCallStructMethod
        self._info = {}

        # IOServiceGetMatchingService consumes the matching dictionary.
        service = iokit.IOServiceGetMatchingService(
            0, iokit.IOServiceMatching(b"AppleSMC"))
        self._connection = u32()
        status = iokit.IOServiceOpen(
            service, u32.in_dll(iokit, "mach_task_self_"), 0,
            ctypes.byref(self._connection))
        iokit.IOObjectRelease(service)
        if not service or status:
            raise NVMLError("failed to open AppleSMC")

    def _call(self, **fields):
        request, response = _SMCKeyData(**fields), _SMCKeyData()
        size = ctypes.c_size_t(ctypes.sizeof(response))
        status = self._call_method(
            self._connection, 2,  # kSMCHandleYPCEvent
            ctypes.byref(request), ctypes.sizeof(request),
            ctypes.byref(response), ctypes.byref(size))
        if status or response.result:
            raise NVMLError("SMC call failed ({}, {})".format(
                status, response.result))
        return response

    def read(self, key):
        """Read a key as a number; None if its type is not numeric."""
        code = _fourcc(key)
        # A key's type and size never change, so look them up once.
        if key not in self._info:
            self._info[key] = self._call(key=code, command=_SMC_GET_KEY_INFO)
        info = self._info[key]
        payload = bytes(self._call(
            key=code, command=_SMC_READ_KEY, data_size=info.data_size).bytes)
        if info.data_type == _fourcc("flt "):
            return struct.unpack("<f", payload[:4])[0]
        if info.data_type == _fourcc("ui32"):
            return struct.unpack(">I", payload[:4])[0]
        return None

    def keys(self):
        return [
            struct.pack(">I", self._call(
                command=_SMC_GET_KEY_FROM_INDEX, index=index).key
            ).decode("latin-1")
            for index in range(self.read("#KEY"))
        ]


def read_temperature_celsius(smc_factory=SMC):
    """Average of the plausible SMC ``Tg*`` GPU sensors; raises NVMLError."""
    global _smc
    if _smc is None:
        smc = smc_factory()
        # The key list never changes, and listing it takes thousands of calls.
        _smc = smc, [key for key in smc.keys() if key.startswith("Tg")]
    smc, keys = _smc

    low, high = TEMPERATURE_RANGE
    values = []
    for key in keys:
        try:
            value = smc.read(key)
        except NVMLError:
            continue  # one unreadable key must not hide the others
        if value is not None and low < value <= high:
            values.append(value)
    if not values:
        raise NVMLError("no SMC GPU temperature sensor")
    return sum(values) / len(values)


def ensure_initialized():
    """Refresh the snapshot used by the NVML-compatible API."""
    devices = query()
    for device in devices:
        try:
            device["temperature"] = read_temperature_celsius()
        except (NVMLError, OSError, AttributeError, ValueError) as exc:
            device["temperature_error"] = str(exc)
    _state.devices = devices


def nvmlDeviceGetCount():
    return len(getattr(_state, "devices", []))


def nvmlDeviceGetHandleByIndex(index):
    for device in getattr(_state, "devices", []):
        if device["index"] == index:
            return device
    raise NVMLError_Unknown("Apple GPU index not found: {}".format(index))


def nvmlDeviceGetIndex(handle):
    return handle["index"]


def nvmlDeviceGetName(handle):
    return handle["name"]


def nvmlDeviceGetUUID(handle):
    return handle["uuid"]


def _require(handle, key):
    value = handle.get(key)
    if value is None:
        raise NVMLError_NotSupported(
            handle.get(key + "_error", "{} is unavailable".format(key)))
    return value


def nvmlDeviceGetTemperature(handle, sensor=NVML_TEMPERATURE_GPU):
    del sensor
    return int(round(_require(handle, "temperature")))


def nvmlDeviceGetUtilizationRates(handle):
    return UtilizationRates(gpu=_require(handle, "utilization"))


def nvmlDeviceGetMemoryInfo(handle):
    return MemoryInfo(total=psutil.virtual_memory().total,
                      used=_require(handle, "memory.used"))


def _not_supported(handle):
    del handle
    raise NVMLError_NotSupported("not supported on Apple GPUs")


nvmlDeviceGetFanSpeed = _not_supported
nvmlDeviceGetEncoderUtilization = _not_supported
nvmlDeviceGetDecoderUtilization = _not_supported
nvmlDeviceGetPowerUsage = _not_supported
nvmlDeviceGetEnforcedPowerLimit = _not_supported


def nvmlDeviceGetComputeRunningProcesses(handle):
    """Metal does not distinguish compute from graphics: all are listed."""
    return handle["processes"]


def nvmlDeviceGetGraphicsRunningProcesses(handle):
    del handle
    return []


def nvmlSystemGetDriverVersion():
    return platform.mac_ver()[0]


def check_driver_nvml_version(driver_version_str):
    del driver_version_str
