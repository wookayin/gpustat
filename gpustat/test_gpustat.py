"""
Unit or integration tests for gpustat
"""
# flake8: ignore=E501

import ctypes
import os
import shlex
import sys
import types
from collections import namedtuple
from io import StringIO
from typing import Any

import psutil
import pytest
from mockito import ANY, mock, unstub, when, when2

import gpustat
from gpustat.nvml import pynvml, pynvml_monkeypatch

MB = 1024 * 1024


def remove_ansi_codes(s):
    import re
    s = re.compile(r'\x1b[^m]*m').sub('', s)
    s = re.compile(r'\x0f').sub('', s)
    return s

# -----------------------------------------------------------------------------

mock_gpu_handles = [types.SimpleNamespace(value='mock-handle-%d' % i, index=i)
                    for i in range(3)]

def _configure_mock(N=pynvml,
                    _scenario_nonexistent_pid=False,  # GH-95
                    _scenario_failing_one_gpu=None,   # GH-125, GH-81
                    ):
    """Define mock behaviour for pynvml and psutil.{Process,virtual_memory}."""

    # without following patch, unhashable NVMLError makes unit test crash
    N.NVMLError.__hash__ = lambda _: 0
    assert issubclass(N.NVMLError, BaseException)

    unstub(N)  # reset all the stubs

    when(N).nvmlInit().thenReturn()
    when(N).nvmlShutdown().thenReturn()
    when(N).nvmlSystemGetDriverVersion().thenReturn('415.27.mock')

    when(N)._nvmlGetFunctionPointer(...).thenCallOriginalImplementation()

    NUM_GPUS = 3
    when(N).nvmlDeviceGetCount().thenReturn(NUM_GPUS)

    def _return_or_raise(v):
        """Return a callable for thenAnswer() to let exceptions re-raised."""
        def _callable(*args, **kwargs):
            del args, kwargs
            if isinstance(v, Exception):
                raise v
            return v
        return _callable

    for i in range(NUM_GPUS):
        handle = mock_gpu_handles[i]
        if _scenario_failing_one_gpu and i == 2:  # see #81, #125
            assert (_scenario_failing_one_gpu is N.NVMLError_Unknown or
                    _scenario_failing_one_gpu is N.NVMLError_GpuIsLost)
            handle = _scenario_failing_one_gpu()  # see 81

        when(N).nvmlDeviceGetHandleByIndex(i)\
            .thenAnswer(_return_or_raise(handle))
        when(N).nvmlDeviceGetIndex(handle)\
            .thenReturn(i)
        when(N).nvmlDeviceGetName(handle)\
            .thenReturn({
                0: 'GeForce GTX TITAN 0',
                1: 'GeForce GTX TITAN 1',
                2: 'GeForce RTX 2',
            }[i].encode())
        when(N).nvmlDeviceGetUUID(handle)\
            .thenReturn({
                0: b'GPU-10fb0fbd-2696-43f3-467f-d280d906a107',
                1: b'GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2',
                2: b'GPU-50205d95-57b6-f541-2bcb-86c09afed564',
            }[i])

        when(N).nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)\
            .thenReturn([80, 36, 71][i])
        when(N).nvmlDeviceGetFanSpeed(handle)\
            .thenReturn([16, 53, 100][i])
        when(N).nvmlDeviceGetPowerUsage(handle)\
            .thenAnswer(_return_or_raise({
                0: 125000, 1: N.NVMLError_NotSupported(), 2: 250000
            }[i]))
        when(N).nvmlDeviceGetEnforcedPowerLimit(handle)\
            .thenAnswer(_return_or_raise({
                0: 250000, 1: 250000, 2: N.NVMLError_NotSupported()
            }[i]))

        # see also: NvidiaDriverMock
        mock_memory_t = namedtuple("Memory_t", ['total', 'used'])  # c_nvmlMemory_t
        when(N).nvmlDeviceGetMemoryInfo(handle)\
            .thenAnswer(_return_or_raise({
                0: mock_memory_t(total=12883853312, used=8000*MB),
                1: mock_memory_t(total=12781551616, used=9000*MB),
                2: mock_memory_t(total=12781551616, used=0),
            }[i]))
        # this mock function assumes <510.39 behavior (#141)
        when(N, strict=False)\
            .nvmlDeviceGetMemoryInfo(handle, version=ANY())\
            .thenRaise(N.NVMLError_FunctionNotFound)

        mock_utilization_t = namedtuple("Utilization_t", ['gpu', 'memory'])
        when(N).nvmlDeviceGetUtilizationRates(handle)\
            .thenAnswer(_return_or_raise({
                0: mock_utilization_t(gpu=76, memory=0),
                1: mock_utilization_t(gpu=0, memory=0),
                2: N.NVMLError_NotSupported(),  # Not Supported
            }[i]))

        when(N).nvmlDeviceGetEncoderUtilization(handle)\
            .thenAnswer(_return_or_raise({
                0: [88, 167000],  # [value, sample_rate]
                1: [0, 167000],   # [value, sample_rate]
                2: N.NVMLError_NotSupported(),  # Not Supported
            }[i]))
        when(N).nvmlDeviceGetDecoderUtilization(handle)\
            .thenAnswer(_return_or_raise({
                0: [67, 167000],  # [value, sample_rate]
                1: [0, 167000],   # [value, sample_rate]
                2: N.NVMLError_NotSupported(),  # Not Supported
            }[i]))

        # running process information: a bit annoying...
        mock_process_t = namedtuple("Process_t", ['pid', 'usedGpuMemory'])

        if _scenario_nonexistent_pid:
            mock_processes_gpu2_erratic = [
                mock_process_t(99999, 9999*MB),
                mock_process_t(99995, 9995*MB),   # see issue #95
            ]
        else:
            mock_processes_gpu2_erratic = N.NVMLError_NotSupported()

        # see NvidiaDriverMock as well
        when(N).nvmlDeviceGetComputeRunningProcesses(handle)\
            .thenAnswer(_return_or_raise({
                0: [mock_process_t(48448, 4000*MB), mock_process_t(153223, 4000*MB)],
                1: [mock_process_t(192453, 3000*MB), mock_process_t(194826, 6000*MB)],
                2: mock_processes_gpu2_erratic,   # Not Supported or non-existent
            }[i]))

        when(N).nvmlDeviceGetGraphicsRunningProcesses(handle)\
            .thenAnswer(_return_or_raise({
                0: [mock_process_t(48448, 4000*MB)],
                1: [],
                2: N.NVMLError_NotSupported(),
            }[i]))

    # for psutil
    mock_pid_map = {   # mock/stub information for psutil...
        48448:  ('user1', 'python', 85.25, 3.1415),
        154213: ('user1', 'caffe', 16.89, 100.00),
        38310:  ('user3', 'python', 26.23, 99.9653),
        153223: ('user2', 'python', 15.25, 0.0000),
        194826: ('user3', 'caffe', 0.0, 12.5236),
        192453: ('user1', 'torch', 123.2, 0.7312),
    }
    assert 99999 not in mock_pid_map, 'scenario_nonexistent_pid'
    assert 99995 not in mock_pid_map, 'scenario_nonexistent_pid (#95)'

    def _MockedProcess(pid):
        if pid not in mock_pid_map:
            if pid == 99995:
                # simulate a bug reported in #95
                raise FileNotFoundError("/proc/99995/stat")
            else:
                # for a process that does not exist, NoSuchProcess is the
                # type of exceptions supposed to be raised by psutil
                raise psutil.NoSuchProcess(pid=pid)
        username, cmdline, cpuutil, memutil = mock_pid_map[pid]
        p: Any = mock(strict=True)   # psutil.Process
        p.username = lambda: username
        p.cmdline = lambda: [cmdline]
        p.cpu_percent = lambda: cpuutil
        p.memory_percent = lambda: memutil
        p.pid = pid
        return p

    when(psutil).Process(...)\
        .thenAnswer(_MockedProcess)
    when(psutil).virtual_memory()\
        .thenReturn(mock_memory_t(total=8589934592, used=0))


MOCK_EXPECTED_OUTPUT_DEFAULT = os.linesep.join("""\
[0] GeForce GTX TITAN 0 | 80°C,  76 % |  8000 / 12287 MB | user1(4000M) user2(4000M)
[1] GeForce GTX TITAN 1 | 36°C,   0 % |  9000 / 12189 MB | user1(3000M) user3(6000M)
[2] GeForce RTX 2       | 71°C,  ?? % |     0 / 12189 MB | (Not Supported)
""".splitlines())  # noqa: E501

MOCK_EXPECTED_OUTPUT_FULL = os.linesep.join("""\
[0] GeForce GTX TITAN 0 | 80°C,  16 %,  76 % (E:  88 %  D:  67 %),  125 / 250 W |  8000 / 12287 MB | user1:python/48448(4000M) user2:python/153223(4000M)
[1] GeForce GTX TITAN 1 | 36°C,  53 %,   0 % (E:   0 %  D:   0 %),   ?? / 250 W |  9000 / 12189 MB | user1:torch/192453(3000M) user3:caffe/194826(6000M)
[2] GeForce RTX 2       | 71°C, 100 %,  ?? % (E:  ?? %  D:  ?? %),  250 /  ?? W |     0 / 12189 MB | (Not Supported)
""".splitlines())  # noqa: E501

MOCK_EXPECTED_OUTPUT_FULL_PROCESS = os.linesep.join("""\
[0] GeForce GTX TITAN 0 | 80°C,  16 %,  76 % (E:  88 %  D:  67 %),  125 / 250 W |  8000 / 12287 MB | user1:python/48448(4000M) user2:python/153223(4000M)
 ├─  48448 (  85%,  257MB): python
 └─ 153223 (  15%,     0B): python
[1] GeForce GTX TITAN 1 | 36°C,  53 %,   0 % (E:   0 %  D:   0 %),   ?? / 250 W |  9000 / 12189 MB | user1:torch/192453(3000M) user3:caffe/194826(6000M)
 ├─ 192453 ( 123%,   59MB): torch
 └─ 194826 (   0%, 1025MB): caffe
[2] GeForce RTX 2       | 71°C, 100 %,  ?? % (E:  ?? %  D:  ?? %),  250 /  ?? W |     0 / 12189 MB | (Not Supported)
""".splitlines())  # noqa: E501

MOCK_EXPECTED_OUTPUT_NO_PROCESSES = os.linesep.join("""\
[0] GeForce GTX TITAN 0 | 80°C,  76 % |  8000 / 12287 MB
[1] GeForce GTX TITAN 1 | 36°C,   0 % |  9000 / 12189 MB
[2] GeForce RTX 2       | 71°C,  ?? % |     0 / 12189 MB
""".splitlines())  # noqa: E501

# -----------------------------------------------------------------------------


@pytest.fixture
def scenario_basic():
    _configure_mock()


@pytest.fixture
def scenario_nonexistent_pid():
    _configure_mock(_scenario_nonexistent_pid=True)


@pytest.fixture
def scenario_failing_one_gpu(request: pytest.FixtureRequest):
    # request.param should be either NVMLError_Unknown or NVMLError_GpuIsLost
    _configure_mock(_scenario_failing_one_gpu=request.param)
    return dict(expected_message={
        pynvml.NVMLError_GpuIsLost: 'GPU is lost',
        pynvml.NVMLError_Unknown: 'Unknown Error',
    }[request.param])


@pytest.fixture
def nvidia_driver_version(request: pytest.FixtureRequest):
    """See NvidiaDriverMock."""

    nvidia_mock: NvidiaDriverMock = request.param
    nvidia_mock(pynvml)

    if nvidia_mock.name.startswith('430'):
        # AssertionError: gpustat will print (Not Supported) in this case
        request.node.add_marker(pytest.mark.xfail(
            reason="nvmlDeviceGetComputeRunningProcesses_v2 does not exist"))

    yield nvidia_mock


class NvidiaDriverMock:
    """Simulate the behavior of nvml's low-level functions according to a
    specific nvidia driver versions, with backward compatibility in concern.
    In all the scenarios, gpustat should work well with a compatible version
    of pynvml installed.

    For what has changed on the nvidia driver side (a non-exhaustive list), see
    https://github.com/NVIDIA/nvidia-settings/blame/main/src/nvml.h
    https://github.com/NVIDIA/nvidia-settings/blame/main/src/libXNVCtrlAttributes/NvCtrlAttributesPrivate.h

    Noteworthy changes of nvml driviers:
        450.66:    nvmlDeviceGetComputeRunningProcesses_v2
        510.39.01: nvmlDeviceGetComputeRunningProcesses_v3  (_v2 removed)
                   nvmlDeviceGetMemoryInfo_v2

    Relevant github issues:
        #107: nvmlDeviceGetComputeRunningProcesses_v2 added
        #141: nvmlDeviceGetMemoryInfo (v1) broken for 510.39.01+
    """
    INSTANCES = []

    def __init__(self, name, **kwargs):
        self.name = name
        self.feat = kwargs

    def __call__(self, N):
        self.mock_processes(N)
        self.mock_memoryinfo(N)

    def mock_processes(self, N):
        when(N).nvmlDeviceGetComputeRunningProcesses(...).thenCallOriginalImplementation()
        when(N).nvmlDeviceGetGraphicsRunningProcesses(...).thenCallOriginalImplementation()
        when(N).nvmlSystemGetDriverVersion().thenReturn(self.name)

        def process_t(pid, usedGpuMemory):
            return pynvml.c_nvmlProcessInfo_t(
                pid=ctypes.c_uint(pid),
                usedGpuMemory=ctypes.c_ulonglong(usedGpuMemory),
            )

        # more low-level mocking for
        # nvmlDeviceGetComputeRunningProcesses_{v2, v3} & c_nvmlProcessInfo_t
        def _nvmlDeviceGetComputeRunningProcesses_v2(handle, c_count, c_procs):
            # handle: SimpleNamespace (see _configure_mock)
            if c_count._obj.value == 0:
                return pynvml.NVML_ERROR_INSUFFICIENT_SIZE
            else:
                c_count._obj.value = 2
                if handle.index == 0:
                    c = process_t(pid=48448, usedGpuMemory=4000*MB); c_procs[0] = c
                    c = process_t(pid=153223, usedGpuMemory=4000*MB); c_procs[1] = c
                elif handle.index == 1:
                    c = process_t(pid=192453, usedGpuMemory=3000*MB); c_procs[0] = c
                    c = process_t(pid=194826, usedGpuMemory=6000*MB); c_procs[1] = c
                else:
                    return pynvml.NVML_ERROR_NOT_SUPPORTED
            return pynvml.NVML_SUCCESS

        def _nvmlDeviceGetGraphicsRunningProcesses_v2(handle, c_count, c_procs):
            if c_count._obj.value == 0:
                return pynvml.NVML_ERROR_INSUFFICIENT_SIZE
            else:
                if handle.index == 0:
                    c_count._obj.value = 1
                    c = process_t(pid=48448, usedGpuMemory=4000*MB); c_procs[0] = c
                elif handle.index == 1:
                    c_count._obj.value = 0
                else:
                    return pynvml.NVML_ERROR_NOT_SUPPORTED
            return pynvml.NVML_SUCCESS

        # Note: N._nvmlGetFunctionPointer might have been monkey-patched,
        # so this mock should decorate the underlying, unwrapped raw function,
        # NOT a monkey-patched version of pynvml._nvmlGetFunctionPointer.
        for v in [1, 2, 3]:
            _v = f'_v{v}' if v != 1 else ''   # backward compatible v3 -> v2
            stub = when2(pynvml_monkeypatch.original_nvmlGetFunctionPointer,
                         f'nvmlDeviceGetComputeRunningProcesses{_v}')
            if v <= self.nvmlDeviceGetComputeRunningProcesses_v:
                stub.thenReturn(_nvmlDeviceGetComputeRunningProcesses_v2)
            else:
                stub.thenRaise(pynvml.NVMLError(pynvml.NVML_ERROR_FUNCTION_NOT_FOUND))

            stub = when2(pynvml_monkeypatch.original_nvmlGetFunctionPointer,
                         f'nvmlDeviceGetGraphicsRunningProcesses{_v}')
            if v <= self.nvmlDeviceGetComputeRunningProcesses_v:
                stub.thenReturn(_nvmlDeviceGetGraphicsRunningProcesses_v2)
            else:
                stub.thenRaise(pynvml.NVMLError(pynvml.NVML_ERROR_FUNCTION_NOT_FOUND))

    def mock_memoryinfo(self, N):
        nvmlMemory_v2 = 0x02000028
        if self.nvmlDeviceGetMemoryInfo_v == 1:
            mock_memory_t = namedtuple(
                "c_nvmlMemory_t",
                ['total', 'used'],
            )
        elif self.nvmlDeviceGetMemoryInfo_v == 2:
            mock_memory_t = namedtuple(
                "c_nvmlMemory_v2_t",
                ['version', 'total', 'reserved', 'free', 'used'],
            )
            mock_memory_t.__new__.__defaults__ = (nvmlMemory_v2, 0, 0, 0, 0)
        else:
            raise NotImplementedError

        # simulates drivers >= 510.39, where memoryinfo v2 is introduced
        if self.nvmlDeviceGetMemoryInfo_v == 2:
            for handle in mock_gpu_handles:
                # a correct API requires version=... parameter
                # this assumes nvidia driver is also recent enough.
                when(pynvml_monkeypatch, strict=False)\
                    .original_nvmlDeviceGetMemoryInfo(handle, version=nvmlMemory_v2)\
                    .thenReturn({
                        0: mock_memory_t(total=12883853312, used=8000*MB),
                        1: mock_memory_t(total=12781551616, used=9000*MB),
                        2: mock_memory_t(total=12781551616, used=0),
                    }[handle.index])
                # simulate #141: without the v2 parameter, gives wrong result
                when(pynvml_monkeypatch)\
                    .original_nvmlDeviceGetMemoryInfo(handle)\
                    .thenReturn({
                        0: mock_memory_t(total=12883853312, used=8099*MB),
                        1: mock_memory_t(total=12781551616, used=9099*MB),
                        2: mock_memory_t(total=12781551616, used=99*MB),
                    }[handle.index])

        else:  # old drivers < 510.39
            for handle in mock_gpu_handles:
                # when pynvml>=11.510, v2 API can be called but can't be used
                when(N, strict=False)\
                    .nvmlDeviceGetMemoryInfo(handle, version=ANY())\
                    .thenRaise(N.NVMLError_FunctionNotFound)
                # The v1 API will give a correct result for the v1 API
                when(N).nvmlDeviceGetMemoryInfo(handle)\
                    .thenReturn({
                        0: mock_memory_t(total=12883853312, used=8000*MB),
                        1: mock_memory_t(total=12781551616, used=9000*MB),
                        2: mock_memory_t(total=12781551616, used=0),
                    }[handle.index])

    def __getattr__(self, k):
        return self.feat[k]

    @property
    def __name__(self):
        return self.name

    def __repr__(self):
        return self.__name__


NvidiaDriverMock.INSTANCES = [
    NvidiaDriverMock('430.xx.xx',
                     nvmlDeviceGetComputeRunningProcesses_v=1,
                     nvmlDeviceGetMemoryInfo_v=1,
                     ),
    NvidiaDriverMock('450.66',
                     nvmlDeviceGetComputeRunningProcesses_v=2,
                     nvmlDeviceGetMemoryInfo_v=1,
                     ),
    NvidiaDriverMock('510.39.01',
                     nvmlDeviceGetComputeRunningProcesses_v=3,
                     nvmlDeviceGetMemoryInfo_v=2,
                     ),
]


# -----------------------------------------------------------------------------


class TestGPUStat(object):
    """A pytest class suite for gpustat."""

    def setup_method(self):
        print("")
        self.maxDiff = 4096

    def teardown_method(self):
        unstub()

    @staticmethod
    def capture_output(*args):
        f = StringIO()
        import contextlib

        with contextlib.redirect_stdout(f):  # requires python 3.4+
            try:
                gpustat.main(*args)
            except SystemExit as e:
                if e.code != 0:
                    raise AssertionError(
                        "Argparse failed (see above error message)")
        return f.getvalue()

    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("nvidia_driver_version",
                             NvidiaDriverMock.INSTANCES, indirect=True)
    def test_new_query_mocked_basic(self, scenario_basic, nvidia_driver_version):
        """A basic functionality test, in a case where everything is normal."""

        gpustats = gpustat.new_query()
        fp = StringIO()
        gpustats.print_formatted(
            fp=fp, no_color=False, show_user=True,
            show_cmd=True, show_full_cmd=True, show_pid=True,
            show_fan_speed=True, show_codec="enc,dec", show_power=True,
        )

        result = fp.getvalue()
        print(result)

        unescaped = remove_ansi_codes(result)
        # remove first line (header)
        unescaped = os.linesep.join(unescaped.splitlines()[1:])

        assert unescaped == MOCK_EXPECTED_OUTPUT_FULL_PROCESS

        # verify gpustat results (not exhaustive yet)
        assert gpustats.driver_version == nvidia_driver_version.name
        g: gpustat.GPUStat = gpustats.gpus[0]
        assert g.memory_used == 8000
        assert g.power_draw == 125
        assert g.utilization == 76
        assert g.processes and g.processes[0]['pid'] == 48448

    def test_new_query_mocked_nonexistent_pid(self, scenario_nonexistent_pid):
        """
        Test a case where nvidia query returns non-existent pids (see #16, #18)
        for GPU index 2.
        """
        fp = StringIO()

        gpustats = gpustat.new_query()
        gpustats.print_formatted(fp=fp)

        ret = fp.getvalue()
        print(ret)

        # gpu 2: should ignore process id
        line = remove_ansi_codes(ret).split('\n')[3]
        assert '[2] GeForce RTX 2' in line, str(line)
        assert '99999' not in line
        assert '(Not Supported)' not in line

    @pytest.mark.parametrize("scenario_failing_one_gpu", [
        pynvml.NVMLError_GpuIsLost,
        pynvml.NVMLError_Unknown,
    ], indirect=True)
    def test_new_query_mocked_failing_one_gpu(self, scenario_failing_one_gpu):
        """Test a case where one GPU is failing (see #125)."""
        fp = StringIO()
        gpustats = gpustat.new_query()
        gpustats.print_formatted(fp=fp, show_header=False)
        ret = fp.getvalue()
        print(ret)

        lines = remove_ansi_codes(ret).split('\n')
        message = scenario_failing_one_gpu['expected_message']

        # gpu 2: failing due to unknown error
        line = lines[2]
        assert '[2] ((' + message + '))' in line, str(line)
        assert '99999' not in line
        assert '?°C,   ? %' in line, str(line)
        assert '? /     ? MB' in line, str(line)

        # other gpus should be displayed normally
        assert '[0] GeForce GTX TITAN 0' in lines[0]
        assert '[1] GeForce GTX TITAN 1' in lines[1]

    def test_attributes_and_items(self, scenario_basic):
        """Test whether each property of `GPUStat` instance is well-defined."""

        g = gpustat.new_query()[1]  # includes N/A
        print("(keys) : %s" % str(g.keys()))
        print(g)

        assert g['name'] == g.entry['name']
        assert g['uuid'] == g.uuid

        with pytest.raises(KeyError):
            g['unknown_key']

        print("uuid : %s" % g.uuid)
        print("name : %s" % g.name)
        print("memory : used %d total %d avail %d" % (
            g.memory_used, g.memory_total, g.memory_available))
        print("temperature : %d" % (g.temperature))
        print("utilization : %s" % (g.utilization))
        print("utilization_enc : %s" % (g.utilization_enc))
        print("utilization_dec : %s" % (g.utilization_dec))

    def test_main(self, scenario_basic):
        """Test whether gpustat.main() works well.
        The behavior is mocked exactly as in test_new_query_mocked().
        """
        sys.argv = ['gpustat']
        gpustat.main()

    def test_args_commandline(self, scenario_basic):
        """Tests the end gpustat CLI."""
        capture_output = self.capture_output

        def _remove_ansi_codes_and_header_line(s):
            unescaped = remove_ansi_codes(s)
            # remove first line (header)
            unescaped = os.linesep.join(unescaped.splitlines()[1:])
            return unescaped

        s = capture_output('gpustat', )
        assert _remove_ansi_codes_and_header_line(s) == MOCK_EXPECTED_OUTPUT_DEFAULT

        s = capture_output('gpustat', '--version')
        assert s.startswith('gpustat ')
        print(s)

        s = capture_output('gpustat', '--no-header')
        assert "[0]" in s.splitlines()[0]

        s = capture_output('gpustat', '-a')  # --show-all
        assert _remove_ansi_codes_and_header_line(s) == MOCK_EXPECTED_OUTPUT_FULL

        s = capture_output('gpustat', '--color')
        assert '\x0f' not in s, "Extra \\x0f found (see issue #32)"
        assert _remove_ansi_codes_and_header_line(s) == MOCK_EXPECTED_OUTPUT_DEFAULT

        s = capture_output('gpustat', '--no-color')
        unescaped = remove_ansi_codes(s)
        assert s == unescaped   # should have no ansi code
        assert _remove_ansi_codes_and_header_line(s) == MOCK_EXPECTED_OUTPUT_DEFAULT

        s = capture_output('gpustat', '--no-processes')
        assert _remove_ansi_codes_and_header_line(s) == MOCK_EXPECTED_OUTPUT_NO_PROCESSES

        s = capture_output('gpustat', '--id', '1,2')
        assert _remove_ansi_codes_and_header_line(s) == \
            os.linesep.join(MOCK_EXPECTED_OUTPUT_DEFAULT.splitlines()[1:3])

    def test_args_commandline_width(self, scenario_basic):
        capture_output = self.capture_output

        # see MOCK_EXPECTED_OUTPUT_DEFAULT
        assert len("GeForce GTX TITAN 0") == 19

        s = capture_output('gpustat', '--gpuname-width', '25')
        print("- Should have width=25")
        print(s)
        assert 'GeForce GTX TITAN 0       |' in remove_ansi_codes(s)
        assert 'GeForce RTX 2             |' in remove_ansi_codes(s)
        #                         ^012345
        #                        19

        # See #47 (since v1.0)
        print("- Should have width=10 (with truncation)")
        s = capture_output('gpustat', '--gpuname-width', '10')
        print(s)
        assert '…X TITAN 0 |' in remove_ansi_codes(s)
        assert '…rce RTX 2 |' in remove_ansi_codes(s)
        #       1234567890

        print("- Should have width=1 (too short)")
        s = capture_output('gpustat', '--gpuname-width', '1')
        print(s)
        assert '… |' in remove_ansi_codes(s)

        print("- Should have width=0: no name displayed.")
        s = capture_output('gpustat', '--gpuname-width', '0')
        print(s)
        assert '[0]  80°C' in remove_ansi_codes(s)

        print("- Invalid inputs")
        with pytest.raises(AssertionError, match="Argparse failed"):
            s = capture_output('gpustat', '--gpuname-width', '-1')
        with pytest.raises(AssertionError, match="Argparse failed"):
            s = capture_output('gpustat', '--gpuname-width', 'None')

    def test_args_commandline_showoptions(self, scenario_basic):
        """Tests gpustat CLI with a variety of --show-xxx options. """

        capture_output = self.capture_output
        print('')

        TEST_OPTS = []
        TEST_OPTS += ['-a', '-c', '-u', '-p', '-e', '-P', '-f']
        TEST_OPTS += [('-e', ''), ('-P', '')]
        TEST_OPTS += [('-e', 'enc,dec'), '-Plimit,draw']
        TEST_OPTS += ['-cup', '-cpu', '-cufP']  # 'cpuePf'

        for opt in TEST_OPTS:
            if isinstance(opt, str):
                opt = [opt]

            print('\x1b[30m\x1b[43m',  # black_on_yellow
                  '$ gpustat ' + ' '.join(shlex.quote(o) for o in opt),
                  '\x1b(B\x1b[m', sep='')
            s = capture_output('gpustat', *opt)

            # TODO: Validate output without hardcoding expected outputs
            print(s)

        # Finally, unknown args
        with pytest.raises(AssertionError):
            capture_output('gpustat', '--unrecognized-args-in-test')

    @pytest.mark.skipif(sys.platform == 'win32', reason="Do not run on Windows")
    def test_no_TERM(self, scenario_basic, monkeypatch):
        """--color should work well even when executed without TERM,
        e.g. ssh localhost gpustat --color"""
        monkeypatch.setenv("TERM", "")

        s = self.capture_output('gpustat', '--color', '--no-header').rstrip()
        print(s)
        assert remove_ansi_codes(s) == MOCK_EXPECTED_OUTPUT_DEFAULT, \
            "wrong gpustat output"

        assert '\x1b[36m' in s, "should contain cyan color code"
        assert '\x0f' not in s, "Extra \\x0f found (see issue #32)"

    def test_json_mocked(self, scenario_basic):
        gpustats = gpustat.new_query()

        fp = StringIO()
        gpustats.print_json(fp=fp)

        import json
        j = json.loads(fp.getvalue())

        from pprint import pprint
        pprint(j)

        assert j['driver_version'] == '415.27.mock'
        assert j['hostname']
        assert j['gpus']


if __name__ == '__main__':
    pytest.main()
