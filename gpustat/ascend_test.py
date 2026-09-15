"""Tests for the Huawei Ascend ``npu-smi`` backend."""

from types import SimpleNamespace

import pytest

from gpustat import ascend
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
    monkeypatch.setattr(ascend, "is_available", lambda: True)
    monkeypatch.setattr(ascend, "_enrich_processes", lambda devices: None)
    monkeypatch.setattr(
        ascend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=NPU_SMI_INFO, stderr="", returncode=0),
    )

    devices, _ = ascend.query([1])
    assert [device["index"] for device in devices] == [1]

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
