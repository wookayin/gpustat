"""Tests for the Huawei Ascend backend."""

import subprocess
import threading
from types import SimpleNamespace

import pytest

from gpustat import ascend, backends, core, nvml
from gpustat.core import GPUStatCollection
from gpustat.nvml import pynvml

NPU_SMI_INFO = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 25.0.rc1.2               Version: 25.0.rc1.2                                           |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2C              | OK            | 118.6       47                0    / 0             |
| 0                         | 0000:6B:02.0  | 6           0    / 0          35738/ 65536         |
+===========================+===============+====================================================+
| 1     910B2C              | OK            | 143.2       49                0    / 0             |
| 0                         | 0000:65:02.0  | 11          0    / 0          35761/ 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 6513          | rayWorkerDict            | 617                     |
| 0       0                 | 33776         | VLLMWorker_TP            | 31747                   |
+===========================+===============+====================================================+
| 1       0                 | 33777         | VLLMWorker_TP            | 31747                   |
+===========================+===============+====================================================+
"""


def _parsed_devices():
    return ascend.parse_npu_smi_output(NPU_SMI_INFO)


def test_parse_npu_smi_info():
    devices, driver_version = _parsed_devices()

    assert driver_version == "25.0.rc1.2"
    assert len(devices) == 2
    assert devices[0] == {
        "index": 0,
        "name": "Ascend 910B2C",
        "uuid": "NPU-0000:6B:02.0",
        "temperature": 47,
        "utilization": 6,
        "power": 118.6,
        "memory.used": 35738,
        "memory.total": 65536,
        "processes": [
            {
                "pid": 6513,
                "command": "rayWorkerDict",
                "memory.used": 617,
            },
            {
                "pid": 33776,
                "command": "VLLMWorker_TP",
                "memory.used": 31747,
            },
        ],
        "health": "OK",
    }
    assert devices[1]["utilization"] == 11
    assert devices[1]["memory.used"] == 35761
    assert devices[1]["processes"][0]["pid"] == 33777


def test_query_filters_device_ids(monkeypatch):
    run_calls = []

    def run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return SimpleNamespace(
            stdout=NPU_SMI_INFO, stderr="", returncode=0)

    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(ascend.subprocess, "run", run)

    devices, _ = ascend.query([1])
    assert [device["index"] for device in devices] == [1]
    assert run_calls[0][0] == (["npu-smi", "info"],)
    assert run_calls[0][1]["check"] is True
    assert run_calls[0][1]["timeout"] == 10
    assert run_calls[0][1]["env"]["LC_ALL"] == "C"

    with pytest.raises(ascend.NVMLError,
                       match="Ascend device index not found: 2"):
        ascend.query([2])


def test_query_reports_command_errors(monkeypatch):
    monkeypatch.setattr(ascend, "is_available", lambda: False)
    with pytest.raises(ascend.NVMLError, match="npu-smi was not found"):
        ascend.query()

    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(
        ascend.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, args[0])),
    )
    with pytest.raises(ascend.NVMLError,
                       match="failed to execute npu-smi info"):
        ascend.query()


def test_query_rejects_output_without_devices(monkeypatch):
    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(
        ascend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="npu-smi Version: 25.0.rc1.2\n",
            stderr="",
            returncode=0,
        ),
    )

    with pytest.raises(ascend.NVMLError,
                       match="returned no Ascend devices"):
        ascend.query()


def test_ascend_nvml_compatible_api(monkeypatch):
    devices, driver_version = _parsed_devices()
    monkeypatch.setattr(
        ascend, "query", lambda ids=None: (devices, driver_version))

    ascend.ensure_initialized()
    handle = ascend.nvmlDeviceGetHandleByIndex(0)
    memory = ascend.nvmlDeviceGetMemoryInfo(handle)
    processes = ascend.nvmlDeviceGetComputeRunningProcesses(handle)

    assert ascend.nvmlDeviceGetCount() == 2
    assert ascend.nvmlDeviceGetIndex(handle) == 0
    assert ascend.nvmlDeviceGetName(handle) == "Ascend 910B2C"
    assert ascend.nvmlDeviceGetUUID(handle) == "NPU-0000:6B:02.0"
    assert ascend.nvmlDeviceGetTemperature(handle) == 47
    assert ascend.nvmlDeviceGetUtilizationRates(handle).gpu == 6
    assert ascend.nvmlDeviceGetPowerUsage(handle) == 118600
    assert memory.used == 35738 * ascend.MB
    assert memory.total == 65536 * ascend.MB
    assert processes[0].pid == 6513
    assert processes[0].usedGpuMemory == 617 * ascend.MB
    assert ascend.nvmlDeviceGetHealth(handle) == "OK"
    assert ascend.nvmlSystemGetDriverVersion() == "25.0.rc1.2"


def test_collection_uses_shared_backend_pipeline(monkeypatch):
    devices, driver_version = _parsed_devices()
    devices[1]["processes"] = []
    GPUStatCollection.global_processes.clear()
    monkeypatch.setattr(
        ascend, "query", lambda ids=None: (devices, driver_version))
    monkeypatch.setattr(
        core.time, "sleep",
        lambda _: pytest.fail("idle devices must not delay the query"),
    )

    stats = GPUStatCollection.new_query(backend="ascend", id="1")

    assert len(stats) == 1
    assert stats.driver_version == "25.0.rc1.2"
    assert stats[0].name == "Ascend 910B2C"
    assert stats[0].memory_used == 35761
    assert stats[0].memory_total == 65536
    assert stats[0].utilization == 11
    assert stats[0].entry["health"] == "OK"


def test_shared_pipeline_enriches_ascend_processes(monkeypatch):
    devices, driver_version = _parsed_devices()
    sleep_calls = []
    GPUStatCollection.global_processes.clear()

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.cpu_percent_calls = 0

        def username(self):
            return "alice"

        def cmdline(self):
            return ["/usr/bin/python", "worker.py"]

        def cpu_percent(self):
            self.cpu_percent_calls += 1
            return 12.5 * self.cpu_percent_calls

        def memory_percent(self):
            return 25.0

    monkeypatch.setattr(
        ascend, "query", lambda ids=None: (devices, driver_version))
    monkeypatch.setattr(core.psutil, "Process", Process)
    monkeypatch.setattr(
        core.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=1024),
    )
    monkeypatch.setattr(core.time, "sleep", sleep_calls.append)

    stats = GPUStatCollection.new_query(backend="ascend")
    process = stats[0].processes[0]
    assert sleep_calls == [0.1]
    assert process["username"] == "alice"
    assert process["command"] == "python"
    assert process["full_command"] == ["/usr/bin/python", "worker.py"]
    assert process["cpu_percent"] == 25.0
    assert process["cpu_memory_usage"] == 256
    assert process["gpu_memory_usage"] == 617


def test_shared_pipeline_keeps_process_when_psutil_denies_access(monkeypatch):
    devices, driver_version = _parsed_devices()
    GPUStatCollection.global_processes.clear()

    def denied(pid):
        raise core.psutil.AccessDenied(pid)

    monkeypatch.setattr(
        ascend, "query", lambda ids=None: (devices, driver_version))
    monkeypatch.setattr(core.psutil, "Process", denied)
    monkeypatch.setattr(core.time, "sleep", lambda _: None)

    stats = GPUStatCollection.new_query(backend="ascend", id="0")

    assert stats[0].processes == [
        {
            "pid": 6513,
            "username": "?",
            "command": "rayWorkerDict",
            "full_command": ["rayWorkerDict"],
            "cpu_percent": 0.0,
            "cpu_memory_usage": 0.0,
            "gpu_memory_usage": 617,
        },
        {
            "pid": 33776,
            "username": "?",
            "command": "VLLMWorker_TP",
            "full_command": ["VLLMWorker_TP"],
            "cpu_percent": 0.0,
            "cpu_memory_usage": 0.0,
            "gpu_memory_usage": 31747,
        },
    ]


def test_shared_pipeline_uses_backend_command_for_empty_cmdline(monkeypatch):
    devices, driver_version = _parsed_devices()
    GPUStatCollection.global_processes.clear()

    process = SimpleNamespace(
        username=lambda: "alice",
        cmdline=lambda: [],
        cpu_percent=lambda: 0.0,
        memory_percent=lambda: 0.0,
    )
    monkeypatch.setattr(
        ascend, "query", lambda ids=None: (devices, driver_version))
    monkeypatch.setattr(core.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(core.time, "sleep", lambda _: None)

    stats = GPUStatCollection.new_query(backend="ascend", id="0")

    assert stats[0].processes[0]["command"] == "rayWorkerDict"
    assert stats[0].processes[0]["full_command"] == ["rayWorkerDict"]


def test_snapshot_is_thread_local(monkeypatch):
    devices, _ = _parsed_devices()
    barrier = threading.Barrier(2)
    results = {}

    def query(ids=None):
        index = int(threading.current_thread().name)
        return [devices[index]], "version-{}".format(index)

    def worker(index):
        try:
            ascend.ensure_initialized()
            barrier.wait()
            handle = ascend.nvmlDeviceGetHandleByIndex(index)
            results[index] = (
                ascend.nvmlDeviceGetIndex(handle),
                ascend.nvmlSystemGetDriverVersion(),
            )
        except Exception as error:
            results[index] = error

    monkeypatch.setattr(ascend, "query", query)
    threads = [
        threading.Thread(target=worker, args=(index,), name=str(index))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {
        0: (0, "version-0"),
        1: (1, "version-1"),
    }


def test_auto_backend_falls_back_to_ascend(monkeypatch):
    devices, driver_version = _parsed_devices()

    def unavailable_nvml():
        raise pynvml.NVMLError_Unknown()

    monkeypatch.setattr(backends.util, "has_AMD", lambda: False)
    monkeypatch.setattr(nvml, "ensure_initialized", unavailable_nvml)
    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(
        ascend, "query", lambda ids=None: (devices, driver_version))

    stats = GPUStatCollection.new_query()
    assert len(stats) == 2
    assert stats[0].name == "Ascend 910B2C"
