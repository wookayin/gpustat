"""Huawei Ascend backend backed by ``npu-smi``."""

import os
import re
import shutil
import subprocess
import threading
from collections import namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

MB = 1024 * 1024
NVML_TEMPERATURE_GPU = 1


class NVMLError(Exception):
    """Base error for the Ascend management backend."""


class NVMLError_Unknown(NVMLError):
    """An unknown Ascend management error occurred."""


class NVMLError_GpuIsLost(NVMLError):
    """An Ascend device is unavailable."""


class NVMLError_NotSupported(NVMLError):
    """The requested metric is not supported."""


MemoryInfo = namedtuple("MemoryInfo", ["total", "used"])
UtilizationRates = namedtuple("UtilizationRates", ["gpu"])
ComputeProcess = namedtuple(
    "ComputeProcess", ["pid", "usedGpuMemory", "command"])

_state = threading.local()


def is_available() -> bool:
    """Return whether the Ascend management command is installed."""
    return shutil.which("npu-smi") is not None


def parse_npu_smi_output(
        output: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse the tabular output of ``npu-smi info``."""
    devices: Dict[int, Dict[str, Any]] = {}
    pending_index = None
    parsing_processes = False
    driver_version = None

    for line in output.splitlines():
        if driver_version is None:
            version_match = re.search(r"\bVersion:\s*(\S+)", line)
            if version_match:
                driver_version = version_match.group(1)

        if "Process id" in line and "Process name" in line:
            parsing_processes = True
            pending_index = None
            continue

        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

        if parsing_processes:
            if len(cells) != 4:
                continue
            device_match = re.fullmatch(r"(\d+)\s+(\d+)", cells[0])
            if device_match is None or not cells[1].isdigit():
                continue
            index = int(device_match.group(1))
            if index not in devices:
                continue
            memory_match = re.search(r"\d+", cells[3])
            if memory_match is None:
                continue
            devices[index]["processes"].append({
                "pid": int(cells[1]),
                "command": cells[2],
                "memory.used": int(memory_match.group(0)),
            })
            continue

        if len(cells) != 3:
            continue

        device_match = re.fullmatch(r"(\d+)\s+(.+)", cells[0])
        metrics_match = re.match(r"([\d.]+|-)\s+([\d.]+|-)", cells[2])
        if (device_match is not None and metrics_match is not None
                and not cells[1].lower().startswith("health")):
            index = int(device_match.group(1))
            power, temperature = metrics_match.groups()
            devices[index] = {
                "index": index,
                "name": "Ascend " + device_match.group(2).strip(),
                "uuid": "",
                "temperature": (
                    int(float(temperature)) if temperature != "-" else None
                ),
                "utilization": None,
                "power": float(power) if power != "-" else None,
                "memory.used": 0,
                "memory.total": 0,
                "processes": [],
                "health": cells[1],
            }
            pending_index = index
            continue

        if pending_index is None or pending_index not in devices:
            continue
        bus_id_match = re.fullmatch(
            r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d",
            cells[1],
        )
        if bus_id_match is None:
            continue
        memory_pairs = re.findall(r"(\d+)\s*/\s*(\d+)", cells[2])
        utilization_match = re.match(r"(\d+)", cells[2])
        if not memory_pairs or utilization_match is None:
            continue

        device = devices[pending_index]
        device["uuid"] = "NPU-" + cells[1].upper()
        device["utilization"] = int(utilization_match.group(1))
        device["memory.used"] = int(memory_pairs[-1][0])
        device["memory.total"] = int(memory_pairs[-1][1])
        pending_index = None

    parsed_devices = [
        device for _, device in sorted(devices.items())
        if device["uuid"]
    ]
    return parsed_devices, driver_version


def query(
        ids: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Query all visible Ascend devices using one ``npu-smi`` invocation."""
    if not is_available():
        raise NVMLError("npu-smi was not found in PATH")

    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
            timeout=10,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NVMLError(
            "failed to execute npu-smi info: {}".format(exc)) from exc

    devices, driver_version = parse_npu_smi_output(result.stdout)
    if not devices:
        raise NVMLError("npu-smi info returned no Ascend devices")

    if ids is not None:
        devices_by_id = {device["index"]: device for device in devices}
        missing = [index for index in ids if index not in devices_by_id]
        if missing:
            raise NVMLError(
                "Ascend device index not found: {}".format(
                    ", ".join(str(index) for index in missing)
                )
            )
        devices = [devices_by_id[index] for index in ids]

    return devices, driver_version


def ensure_initialized():
    """Refresh the snapshot used by the NVML-compatible API."""
    _state.devices, _state.driver_version = query()


def nvmlDeviceGetCount():
    return len(getattr(_state, "devices", []))


def nvmlDeviceGetHandleByIndex(index):
    for device in getattr(_state, "devices", []):
        if device["index"] == index:
            return device
    raise NVMLError_Unknown("Ascend device index not found: {}".format(index))


def nvmlDeviceGetIndex(handle):
    return handle["index"]


def nvmlDeviceGetName(handle):
    return handle["name"]


def nvmlDeviceGetUUID(handle):
    return handle["uuid"]


def nvmlDeviceGetTemperature(handle, sensor=NVML_TEMPERATURE_GPU):
    del sensor
    return handle["temperature"]


def nvmlDeviceGetFanSpeed(handle):
    del handle
    return None


def nvmlDeviceGetMemoryInfo(handle):
    return MemoryInfo(
        total=handle["memory.total"] * MB,
        used=handle["memory.used"] * MB,
    )


def nvmlDeviceGetUtilizationRates(handle):
    return UtilizationRates(gpu=handle["utilization"])


def nvmlDeviceGetEncoderUtilization(handle):
    del handle
    return None


def nvmlDeviceGetDecoderUtilization(handle):
    del handle
    return None


def nvmlDeviceGetPowerUsage(handle):
    power = handle["power"]
    return int(round(power * 1000)) if power is not None else None


def nvmlDeviceGetEnforcedPowerLimit(handle):
    del handle
    return None


def nvmlDeviceGetComputeRunningProcesses(handle):
    return [
        ComputeProcess(
            pid=process["pid"],
            usedGpuMemory=process["memory.used"] * MB,
            command=process["command"],
        )
        for process in handle["processes"]
    ]


def nvmlDeviceGetGraphicsRunningProcesses(handle):
    del handle
    return []


def nvmlDeviceGetHealth(handle):
    return handle["health"]


def nvmlSystemGetDriverVersion():
    return getattr(_state, "driver_version", None) or ""


def check_driver_nvml_version(driver_version_str):
    del driver_version_str
