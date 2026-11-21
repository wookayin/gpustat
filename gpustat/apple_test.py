"""Tests for the Apple Silicon GPU backend."""

import ctypes
import io
import json
import platform
import plistlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpustat import apple, ascend, backends, core, nvml
from gpustat.core import GPUStatCollection
from gpustat.nvml import pynvml

MB = 1024 * 1024
NOW_NS = 10000000000

# Trimmed from `ioreg -a -r -l -d 2 -c IOAccelerator` on an M4 Pro.
M4_PRO_ENTRY = {
    "IOClass": "AGXAcceleratorG16X",
    "IORegistryEntryID": 4294968401,
    "model": "Apple M4 Pro",
    "PerformanceStatistics": {
        "Device Utilization %": 27,
        "In use system memory": 508592128,
        "Alloc system memory": 5358436352,
    },
    "IORegistryEntryChildren": [
        {"IOUserClientCreator": "pid 404, WindowServer", "AppUsage": [
            {"API": "Metal", "lastSubmittedTime": 9500000000},
            {"API": "Metal", "lastSubmittedTime": 0},
        ]},
        {"IOUserClientCreator": "pid 404, WindowServer", "AppUsage": []},
        # Submitted too long ago, never submitted, or not a process.
        {"IOUserClientCreator": "pid 761, ControlCenter", "AppUsage": [
            {"API": "Metal", "lastSubmittedTime": 1000000000},
        ]},
        {"IOUserClientCreator": "pid 900, Idle", "AppUsage": []},
        {"IOUserClientCreator": "kernel"},
        {"IOObjectClass": "AGXSharedUserClient"},
        {"IOUserClientCreator": "pid 78129, python3", "AppUsage": [
            {"API": "Metal", "lastSubmittedTime": 9900000000},
        ]},
    ],
}
INTEL_ENTRY = {"IOClass": "IntelAccelerator", "model": "Intel Iris"}


@pytest.fixture(autouse=True)
def fake_host(monkeypatch):
    """Keep every test away from the real ioreg, SMC, and clock."""
    monkeypatch.setattr(apple, "is_available", lambda: True)
    monkeypatch.setattr(apple, "_smc", None)
    monkeypatch.setattr(apple, "read_temperature_celsius", lambda: 48.4)
    monkeypatch.setattr(
        apple.time, "clock_gettime_ns", lambda clock: NOW_NS, raising=False)
    monkeypatch.setattr(apple.platform, "mac_ver", lambda: ("27.2", "", ""))
    monkeypatch.setattr(
        apple.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=24 * 1024 * MB))
    _fake_ioreg(monkeypatch, plistlib.dumps([INTEL_ENTRY, M4_PRO_ENTRY]))
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    monkeypatch.setattr(core.psutil, "Process", FakeProcess)
    GPUStatCollection.global_processes.clear()
    yield
    GPUStatCollection.global_processes.clear()


def _fake_ioreg(monkeypatch, output=b"", error=None):
    def run(command, **kwargs):
        assert command == apple.IOREG_COMMAND
        if error:
            raise error
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(apple.subprocess, "run", run)


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def username(self):
        return "user"

    def cmdline(self):
        return ["/usr/bin/python3", "train.py"]

    def cpu_percent(self):
        return 3.0

    def memory_percent(self):
        return 1.0


def test_query():
    assert apple.query() == [{
        "index": 0,
        "name": "Apple M4 Pro",
        "uuid": "APPLE-GPU-4294968401",
        "utilization": 27,
        "memory.used": 508592128,
        "processes": [(404, "WindowServer"), (78129, "python3")],
    }]


@pytest.mark.parametrize("output, error, message", [
    (b"", None, "no Apple GPU"),
    (plistlib.dumps([INTEL_ENTRY]), None, "no Apple GPU"),
    (b"not a plist", None, "failed to query ioreg"),
    (b"", OSError("ioreg missing"), "failed to query ioreg"),
    (b"", subprocess.TimeoutExpired("ioreg", 10), "failed to query ioreg"),
])
def test_query_errors(monkeypatch, output, error, message):
    _fake_ioreg(monkeypatch, output, error)

    with pytest.raises(apple.NVMLError, match=message):
        apple.query()


def test_missing_statistics_are_unsupported(monkeypatch):
    _fake_ioreg(monkeypatch, plistlib.dumps([{"IOClass": "AGXAccelerator"}]))

    gpu = GPUStatCollection.new_query(backend="apple")[0]

    assert gpu.name == "Apple GPU"
    assert gpu.utilization is None
    assert gpu.memory_used is None


def test_smc_key_data_matches_kernel_layout():
    assert ctypes.sizeof(apple._SMCKeyData) == 80
    assert apple._SMCKeyData.data_size.offset == 28
    assert apple._SMCKeyData.result.offset == 40
    assert apple._SMCKeyData.command.offset == 42
    assert apple._SMCKeyData.bytes.offset == 48


class FakeSMC:
    def __init__(self, values):
        self.values = values
        self.key_listings = 0

    def keys(self):
        self.key_listings += 1
        return ["#KEY", "TC0P"] + list(self.values)

    def read(self, key):
        if isinstance(self.values[key], Exception):
            raise self.values[key]
        return self.values[key]


def test_temperature_averages_plausible_smc_gpu_keys(monkeypatch):
    monkeypatch.undo()  # restore the real read_temperature_celsius
    monkeypatch.setattr(apple, "_smc", None)
    smc = FakeSMC({
        "Tg05": 40.0, "Tg0K": 50.0, "Tg0L": None, "Tg0R": 4.0, "Tg0S": 151.0,
        "Tg0X": apple.NVMLError("SMC call failed"),
    })

    assert apple.read_temperature_celsius(lambda: smc) == 45.0
    assert apple.read_temperature_celsius(lambda: smc) == 45.0
    assert smc.key_listings == 1

    smc.values = dict.fromkeys(smc.values, 0.0)
    with pytest.raises(apple.NVMLError, match="no SMC GPU temperature"):
        apple.read_temperature_celsius()


def test_collection(monkeypatch):
    stats = GPUStatCollection.new_query(backend="apple")
    text, data = io.StringIO(), io.StringIO()
    stats.print_formatted(fp=text, no_color=True, show_cmd=True,
                          show_user=True, show_pid=True)
    stats.print_json(fp=data)

    assert stats.driver_version == "27.2"
    assert text.getvalue().splitlines()[1] == (
        "[0] Apple M4 Pro | 48°C,  27 % |   485 / 24576 MB |"
        " user:python3/404 user:python3/78129")
    gpu = json.loads(data.getvalue())["gpus"][0]
    assert gpu["uuid"] == "APPLE-GPU-4294968401"
    assert gpu["memory.unified"] is True
    assert gpu["power.draw"] is None
    assert gpu["fan.speed"] is None
    assert [process["pid"] for process in gpu["processes"]] == [404, 78129]
    assert "gpu_memory_usage" not in gpu["processes"][0]


def test_temperature_errors_are_unsupported(monkeypatch):
    def broken():
        raise apple.NVMLError("failed to open AppleSMC")

    monkeypatch.setattr(apple, "read_temperature_celsius", broken)

    stats = GPUStatCollection.new_query(backend="apple")

    assert stats[0].temperature is None
    assert stats[0].utilization == 27
    with pytest.raises(apple.NVMLError_NotSupported, match="AppleSMC"):
        apple.nvmlDeviceGetTemperature(apple.nvmlDeviceGetHandleByIndex(0))


def test_collection_rejects_unknown_index():
    stats = GPUStatCollection.new_query(backend="apple", id="1")

    assert isinstance(stats[0], core.InvalidGPU)


def _without_other_backends(monkeypatch):
    def unavailable_nvml():
        raise pynvml.NVMLError_LibraryNotFound()

    monkeypatch.setattr(backends.util, "has_AMD", lambda: False)
    monkeypatch.setattr(nvml, "ensure_initialized", unavailable_nvml)
    monkeypatch.setattr(ascend, "is_available", lambda: False)


def test_auto_backend_falls_back_to_apple(monkeypatch):
    _without_other_backends(monkeypatch)

    assert backends.detect_backend().name == "apple"


def test_auto_backend_keeps_nvml_error_without_apple_gpu(monkeypatch):
    _without_other_backends(monkeypatch)
    _fake_ioreg(monkeypatch, plistlib.dumps([INTEL_ENTRY]))

    with pytest.raises(pynvml.NVMLError_LibraryNotFound):
        backends.detect_backend()


def test_auto_backend_skips_apple_off_macos(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(apple.sys, "platform", "linux")

    assert backends._detect_apple() is False


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="requires an Apple Silicon GPU")
def test_hardware(monkeypatch):
    monkeypatch.undo()
    try:
        apple.ensure_initialized()
    except apple.NVMLError:
        pytest.skip("no AGX GPU, e.g. a CI virtual machine")
    handle = apple.nvmlDeviceGetHandleByIndex(0)

    assert 0 <= apple.nvmlDeviceGetUtilizationRates(handle).gpu <= 100
    assert apple.nvmlDeviceGetMemoryInfo(handle).used > 0
    assert 10 < apple.nvmlDeviceGetTemperature(handle) <= 150
