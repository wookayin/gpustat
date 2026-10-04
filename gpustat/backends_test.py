"""Tests for accelerator built-in backend selection."""

from types import SimpleNamespace

import pytest

from gpustat import backends, core, nvml
from gpustat.core import GPUStatCollection
from gpustat.nvml import pynvml


def test_auto_backend_prefers_nvidia(monkeypatch):
    monkeypatch.setattr(backends.util, "has_AMD", lambda: False)
    monkeypatch.setattr(nvml, "ensure_initialized", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 1)

    assert backends.detect_backend().name == "nvidia"


def test_auto_backend_preserves_amd_first_detection(monkeypatch):
    expected = backends.Backend(
        "amd", SimpleNamespace(ensure_initialized=lambda: None),
        SimpleNamespace(nvmlDeviceGetCount=lambda: 1), lambda _: None)
    monkeypatch.setattr(backends.util, "has_AMD", lambda: True)
    monkeypatch.setitem(backends._LOADED, "amd", expected)

    assert backends.detect_backend() is expected


def test_gpu_count_uses_selected_backend(monkeypatch):
    expected = backends.Backend(
        "amd", SimpleNamespace(ensure_initialized=lambda: None),
        SimpleNamespace(nvmlDeviceGetCount=lambda: 4), lambda _: None)
    monkeypatch.setattr(core, "get_backend", lambda name: expected)

    assert core.gpu_count(backend="amd") == 4


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend: tpu"):
        GPUStatCollection.new_query(backend="tpu")
    with pytest.raises(ValueError, match="Unknown backend: tpu"):
        core.gpu_count(backend="tpu")
