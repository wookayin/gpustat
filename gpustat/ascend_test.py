"""Tests for the Huawei Ascend ``npu-smi`` backend."""

import subprocess
from types import SimpleNamespace

import psutil
import pytest

from gpustat import ascend
from gpustat import core
from gpustat.core import GPUStatCollection


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


def test_parse_npu_smi_info():
    devices, driver_version = ascend.parse_npu_smi_output(
        NPU_SMI_INFO, enrich_processes=False)

    assert driver_version == "25.0.rc1.2"
    assert len(devices) == 2
    assert devices[0] == {
        "index": 0,
        "name": "Ascend 910B2C",
        "uuid": "NPU-0000:6B:02.0",
        "temperature.gpu": 47,
        "fan.speed": None,
        "utilization.gpu": 6,
        "utilization.enc": None,
        "utilization.dec": None,
        "power.draw": 118,
        "enforced.power.limit": None,
        "memory.used": 35738,
        "memory.total": 65536,
        "processes": [
            {
                "pid": 6513,
                "npu-smi.command": "rayWorkerDict",
                "gpu_memory_usage": 617,
                "gpu_uuid": "NPU-0000:6B:02.0",
            },
            {
                "pid": 33776,
                "npu-smi.command": "VLLMWorker_TP",
                "gpu_memory_usage": 31747,
                "gpu_uuid": "NPU-0000:6B:02.0",
            },
        ],
        "health": "OK",
    }
    assert devices[1]["utilization.gpu"] == 11
    assert devices[1]["memory.used"] == 35761
    assert devices[1]["processes"][0]["pid"] == 33777


def test_query_filters_device_ids(monkeypatch):
    run_calls = []

    def run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return SimpleNamespace(
            stdout=NPU_SMI_INFO, stderr="", returncode=0)

    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(ascend, "_enrich_processes", lambda devices: None)
    monkeypatch.setattr(ascend.subprocess, "run", run)

    devices, _ = ascend.query([1])
    assert [device["index"] for device in devices] == [1]
    assert run_calls[0][0] == (["npu-smi", "info"],)
    assert run_calls[0][1]["check"] is True
    assert run_calls[0][1]["timeout"] == 10
    assert run_calls[0][1]["env"]["LC_ALL"] == "C"

    with pytest.raises(ascend.AscendSmiError,
                       match="Ascend device index not found: 2"):
        ascend.query([2])


def test_parse_empty_output():
    devices, driver_version = ascend.parse_npu_smi_output(
        "", enrich_processes=False)
    assert devices == []
    assert driver_version is None


def test_collection_uses_ascend_backend(monkeypatch):
    devices, driver_version = ascend.parse_npu_smi_output(
        NPU_SMI_INFO, enrich_processes=False)
    queried_ids = []

    def query(ids):
        queried_ids.append(ids)
        return devices[1:], driver_version

    monkeypatch.setattr(ascend, "query", query)
    stats = GPUStatCollection.new_query(backend="ascend", id="1")

    assert queried_ids == [[1]]
    assert stats.driver_version == "25.0.rc1.2"
    assert stats[0].name == "Ascend 910B2C"


def test_query_reports_command_errors(monkeypatch):
    monkeypatch.setattr(ascend, "is_available", lambda: False)
    with pytest.raises(ascend.AscendSmiError,
                       match="npu-smi was not found"):
        ascend.query()

    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(
        ascend.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, args[0])),
    )
    with pytest.raises(ascend.AscendSmiError,
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

    with pytest.raises(ascend.AscendSmiError,
                       match="returned no Ascend devices"):
        ascend.query()


def test_enrich_processes(monkeypatch):
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

    processes = {}

    def make_process(pid):
        processes[pid] = Process(pid)
        return processes[pid]

    monkeypatch.setattr(ascend.psutil, "Process", make_process)
    monkeypatch.setattr(
        ascend.psutil, "virtual_memory",
        lambda: SimpleNamespace(total=1024),
    )
    monkeypatch.setattr(ascend.time, "sleep", lambda _: None)

    devices, _ = ascend.parse_npu_smi_output(NPU_SMI_INFO)
    process = devices[0]["processes"][0]
    assert process["username"] == "alice"
    assert process["command"] == "python"
    assert process["full_command"] == ["/usr/bin/python", "worker.py"]
    assert process["cpu_percent"] == 25.0
    assert process["cpu_memory_usage"] == 256


def test_enrich_processes_uses_npu_smi_name_for_missing_pid(monkeypatch):
    def missing_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(ascend.psutil, "Process", missing_process)
    monkeypatch.setattr(ascend.time, "sleep", lambda _: None)

    devices, _ = ascend.parse_npu_smi_output(NPU_SMI_INFO)
    process = devices[0]["processes"][0]
    assert process["username"] == "?"
    assert process["command"] == "rayWorkerDict"
    assert process["full_command"] == ["rayWorkerDict"]
    assert process["cpu_percent"] == 0.0
    assert process["cpu_memory_usage"] == 0.0


def test_auto_backend_falls_back_to_ascend(monkeypatch):
    devices, driver_version = ascend.parse_npu_smi_output(
        NPU_SMI_INFO, enrich_processes=False)

    def unavailable_nvml():
        raise core.N.NVMLError_Unknown()

    monkeypatch.setattr(core.nvml, "ensure_initialized", unavailable_nvml)
    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(
        ascend, "query", lambda ids: (devices, driver_version))

    stats = GPUStatCollection.new_query()
    assert len(stats) == 2
    assert stats[0].name == "Ascend 910B2C"


def test_auto_backend_prefers_nvidia(monkeypatch):
    expected = object()
    monkeypatch.setattr(core.nvml, "ensure_initialized", lambda: None)
    monkeypatch.setattr(core.N, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(
        GPUStatCollection,
        "_new_query_nvidia",
        lambda debug=False, id=None: expected,
    )
    monkeypatch.setattr(
        ascend, "is_available",
        lambda: pytest.fail("Ascend should not be queried"),
    )

    assert GPUStatCollection.new_query() is expected


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend: tpu"):
        GPUStatCollection.new_query(backend="tpu")
    with pytest.raises(ValueError, match="Unknown backend: tpu"):
        core.gpu_count(backend="tpu")
