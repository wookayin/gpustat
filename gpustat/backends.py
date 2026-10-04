"""Accelerator backend registry.

Backends expose a small NVML-compatible API so the core collection and
formatting logic remains vendor-neutral.
"""

from typing import Dict, List, Tuple

from gpustat import util


class Backend:
    """Adapter for a built-in accelerator management implementation."""

    def __init__(self, name, lifecycle, api, check_driver_version):
        self.name = name
        self.lifecycle = lifecycle
        self.api = api
        self.check_driver_version = check_driver_version

    def ensure_initialized(self):
        self.lifecycle.ensure_initialized()

    def device_count(self):
        return self.api.nvmlDeviceGetCount()


_LOADED: Dict[str, Backend] = {}


def backend_names(include_auto=False) -> Tuple[str, ...]:
    """Return built-in backend names in auto-detection order."""
    names = tuple(name for name, _, _ in _BACKENDS)
    return ('auto',) + names if include_auto else names


def get_backend(name: str) -> Backend:
    """Load a built-in backend by name."""
    loaders = {backend_name: loader for backend_name, loader, _ in _BACKENDS}
    if name not in loaders:
        raise ValueError("Unknown backend: {}".format(name))
    if name not in _LOADED:
        _LOADED[name] = loaders[name]()
    return _LOADED[name]


def detect_backend() -> Backend:
    """Return the first available backend."""
    errors: List[Exception] = []
    for name, _, detector in _BACKENDS:
        try:
            if detector():
                return get_backend(name)
        except Exception as error:
            errors.append(error)

    if errors:
        raise errors[0]
    return get_backend('nvidia')


def _load_nvidia():
    from gpustat import nvml
    from gpustat.nvml import check_driver_nvml_version, pynvml
    return Backend(
        'nvidia', nvml, pynvml, check_driver_nvml_version)


def _detect_nvidia():
    backend = get_backend('nvidia')
    backend.ensure_initialized()
    return backend.device_count() > 0


def _load_amd():
    from gpustat import rocml
    return Backend('amd', rocml, rocml, rocml.check_driver_nvml_version)


def _detect_amd():
    return util.has_AMD()


def _load_ascend():
    from gpustat import ascend
    return Backend(
        'ascend', ascend, ascend, ascend.check_driver_nvml_version)


def _detect_ascend():
    from gpustat import ascend
    return ascend.is_available()


_BACKENDS = (
    ('amd', _load_amd, _detect_amd),
    ('nvidia', _load_nvidia, _detect_nvidia),
    ('ascend', _load_ascend, _detect_ascend),
)
