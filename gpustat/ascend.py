"""Huawei Ascend device queries backed by ``npu-smi``."""

import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil


class AscendSmiError(RuntimeError):
    """Raised when ``npu-smi`` cannot provide a device snapshot."""


def is_available() -> bool:
    """Return whether the Ascend management command is installed."""
    return shutil.which("npu-smi") is not None


def _safe_process_call(fn, default):
    try:
        return fn()
    except (psutil.AccessDenied, psutil.NoSuchProcess, FileNotFoundError):
        return default


def _enrich_processes(devices: List[Dict[str, Any]]) -> None:
    processes = {}
    for device in devices:
        for process in device["processes"]:
            pid = process["pid"]
            try:
                ps_process = psutil.Process(pid)
            except (psutil.AccessDenied, psutil.NoSuchProcess, FileNotFoundError):
                process.update(
                    username="?",
                    command=process["npu-smi.command"],
                    full_command=[process["npu-smi.command"]],
                    cpu_percent=0.0,
                    cpu_memory_usage=0.0,
                )
                continue

            processes[pid] = ps_process
            process["username"] = _safe_process_call(ps_process.username, "?")
            command = _safe_process_call(ps_process.cmdline, [])
            if command:
                process["command"] = os.path.basename(command[0])
                process["full_command"] = command
            else:
                process["command"] = process["npu-smi.command"]
                process["full_command"] = [process["npu-smi.command"]]
            process["cpu_percent"] = _safe_process_call(
                ps_process.cpu_percent, 0.0)
            process["cpu_memory_usage"] = _safe_process_call(
                lambda: round(
                    ps_process.memory_percent() / 100.0
                    * psutil.virtual_memory().total
                ),
                0.0,
            )

    if processes:
        time.sleep(0.1)
        for device in devices:
            for process in device["processes"]:
                ps_process = processes.get(process["pid"])
                if ps_process is not None:
                    process["cpu_percent"] = _safe_process_call(
                        ps_process.cpu_percent, 0.0)


def parse_npu_smi_output(
        output: str, enrich_processes: bool = True
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
                "npu-smi.command": cells[2],
                "gpu_memory_usage": int(memory_match.group(0)),
                "gpu_uuid": devices[index]["uuid"],
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
                "temperature.gpu": (
                    int(float(temperature)) if temperature != "-" else None
                ),
                "fan.speed": None,
                "utilization.gpu": None,
                "utilization.enc": None,
                "utilization.dec": None,
                "power.draw": int(float(power)) if power != "-" else None,
                "enforced.power.limit": None,
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
        device["utilization.gpu"] = int(utilization_match.group(1))
        device["memory.used"] = int(memory_pairs[-1][0])
        device["memory.total"] = int(memory_pairs[-1][1])
        pending_index = None

    parsed_devices = [
        device for _, device in sorted(devices.items())
        if device["uuid"]
    ]
    if enrich_processes:
        _enrich_processes(parsed_devices)
    return parsed_devices, driver_version


def query(
        ids: Optional[Sequence[int]] = None,
        enrich_processes: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Query all visible Ascend devices using one ``npu-smi`` invocation."""
    if not is_available():
        raise AscendSmiError("npu-smi was not found in PATH")

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
        raise AscendSmiError("failed to execute npu-smi info: {}".format(
            exc)) from exc

    devices, driver_version = parse_npu_smi_output(
        result.stdout, enrich_processes=False)
    if not devices:
        raise AscendSmiError("npu-smi info returned no Ascend devices")

    if ids is not None:
        devices_by_id = {device["index"]: device for device in devices}
        missing = [index for index in ids if index not in devices_by_id]
        if missing:
            raise AscendSmiError(
                "Ascend device index not found: {}".format(
                    ", ".join(str(index) for index in missing)
                )
            )
        devices = [devices_by_id[index] for index in ids]

    if enrich_processes:
        _enrich_processes(devices)
    return devices, driver_version
