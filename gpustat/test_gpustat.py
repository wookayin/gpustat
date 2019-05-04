"""
Unit or integration tests for gpustat
"""
# flake8: ignore=E501

from __future__ import print_function
from __future__ import absolute_import

import unittest
import sys
from collections import namedtuple

import psutil
import pynvml
from six.moves import cStringIO as StringIO

import gpustat

try:
    import unittest.mock as mock
except ImportError:
    import mock

MagicMock = mock.MagicMock


def _configure_mock(N, Process,
                    scenario_nonexistent_pid=False):
    """
    Define mock behaviour for N: the pynvml module, and psutil.Process,
    which should be MagicMock objects from unittest.mock.
    """

    # Restore some non-mock objects (such as exceptions)
    for attr in dir(pynvml):
        if attr.startswith('NVML'):
            setattr(N, attr, getattr(pynvml, attr))
    assert issubclass(N.NVMLError, BaseException)

    # without following patch, unhashable NVMLError distrubs unit test
    N.NVMLError.__hash__ = lambda _: 0

    # mock-patch every nvml**** functions used in gpustat.
    N.nvmlInit = MagicMock()
    N.nvmlShutdown = MagicMock()
    N.nvmlDeviceGetCount.return_value = 3
    N.nvmlSystemGetDriverVersion.return_value = '415.27.mock'

    mock_handles = ['mock-handle-%d' % i for i in range(3)]

    def _raise_ex(fn):
        """Decorator to let exceptions returned from the callable re-throwed."""  # noqa:E501
        def _decorated(*args, **kwargs):
            v = fn(*args, **kwargs)
            if isinstance(v, Exception):
                raise v
            return v
        return _decorated

    N.nvmlDeviceGetHandleByIndex.side_effect = \
        lambda index: mock_handles[index]
    N.nvmlDeviceGetIndex.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: 0,
        mock_handles[1]: 1,
        mock_handles[2]: 2,
    }.get(handle, RuntimeError))
    N.nvmlDeviceGetName.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: b'GeForce GTX TITAN 0',
        mock_handles[1]: b'GeForce GTX TITAN 1',
        mock_handles[2]: b'GeForce GTX TITAN 2',
    }.get(handle, RuntimeError))
    N.nvmlDeviceGetUUID.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: b'GPU-10fb0fbd-2696-43f3-467f-d280d906a107',
        mock_handles[1]: b'GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2',
        mock_handles[2]: b'GPU-50205d95-57b6-f541-2bcb-86c09afed564',
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetTemperature = _raise_ex(lambda handle, _: {
        mock_handles[0]: 80,
        mock_handles[1]: 36,
        mock_handles[2]: 71,
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetFanSpeed = _raise_ex(lambda handle: {
        mock_handles[0]: 16,
        mock_handles[1]: 53,
        mock_handles[2]: 100,
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetPowerUsage = _raise_ex(lambda handle: {
        mock_handles[0]: 125000,
        mock_handles[1]: N.NVMLError_NotSupported(),  # Not Supported
        mock_handles[2]: 250000,
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetEnforcedPowerLimit = _raise_ex(lambda handle: {
        mock_handles[0]: 250000,
        mock_handles[1]: 250000,
        mock_handles[2]: N.NVMLError_NotSupported(),  # Not Supported
    }.get(handle, RuntimeError))

    mock_memory_t = namedtuple("Memory_t", ['total', 'used'])
    N.nvmlDeviceGetMemoryInfo.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: mock_memory_t(total=12883853312, used=8000*MB),
        mock_handles[1]: mock_memory_t(total=12781551616, used=9000*MB),
        mock_handles[2]: mock_memory_t(total=12781551616, used=0),
    }.get(handle, RuntimeError))

    mock_utilization_t = namedtuple("Utilization_t", ['gpu', 'memory'])
    N.nvmlDeviceGetUtilizationRates.side_effect = _raise_ex(lambda handle: {
        mock_handles[0]: mock_utilization_t(gpu=76, memory=0),
        mock_handles[1]: mock_utilization_t(gpu=0, memory=0),
        mock_handles[2]: N.NVMLError_NotSupported(),  # Not Supported
    }.get(handle, RuntimeError))

    # running process information: a bit annoying...
    mock_process_t = namedtuple("Process_t", ['pid', 'usedGpuMemory'])

    if scenario_nonexistent_pid:
        mock_processes_gpu2_erratic = [mock_process_t(99999, 9999*MB)]
    else:
        mock_processes_gpu2_erratic = N.NVMLError_NotSupported()
    N.nvmlDeviceGetComputeRunningProcesses.side_effect = _raise_ex(lambda handle: {  # noqa: E501
        mock_handles[0]: [mock_process_t(48448, 4000*MB), mock_process_t(153223, 4000*MB)],  # noqa: E501
        mock_handles[1]: [mock_process_t(192453, 3000*MB), mock_process_t(194826, 6000*MB)],  # noqa: E501
        # Not Supported or non-existent
        mock_handles[2]: mock_processes_gpu2_erratic,
    }.get(handle, RuntimeError))

    N.nvmlDeviceGetGraphicsRunningProcesses.side_effect = _raise_ex(lambda handle: {  # noqa: E501
        mock_handles[0]: [],
        mock_handles[1]: [],
        mock_handles[2]: N.NVMLError_NotSupported(),
    }.get(handle, RuntimeError))

    mock_pid_map = {   # mock information for psutil...
        48448:  ('user1', 'python'),
        154213: ('user1', 'caffe'),
        38310:  ('user3', 'python'),
        153223: ('user2', 'python'),
        194826: ('user3', 'caffe'),
        192453: ('user1', 'torch'),
    }

    def _MockedProcess(pid):
        if pid not in mock_pid_map:
            raise psutil.NoSuchProcess(pid=pid)
        username, cmdline = mock_pid_map[pid]
        p = MagicMock()  # mocked process
        p.username.return_value = username
        p.cmdline.return_value = [cmdline]
        return p
    Process.side_effect = _MockedProcess


MOCK_EXPECTED_OUTPUT_DEFAULT = """\
[0] GeForce GTX TITAN 0 | 80'C,  76 % |  8000 / 12287 MB | user1(4000M) user2(4000M)
[1] GeForce GTX TITAN 1 | 36'C,   0 % |  9000 / 12189 MB | user1(3000M) user3(6000M)
[2] GeForce GTX TITAN 2 | 71'C,  ?? % |     0 / 12189 MB | (Not Supported)
"""  # noqa: E501

MOCK_EXPECTED_OUTPUT_FULL = """\
[0] GeForce GTX TITAN 0 | 80'C,  16 %,  76 %,  125 / 250 W |  8000 / 12287 MB | user1:python/48448(4000M) user2:python/153223(4000M)
[1] GeForce GTX TITAN 1 | 36'C,  53 %,   0 %,   ?? / 250 W |  9000 / 12189 MB | user1:torch/192453(3000M) user3:caffe/194826(6000M)
[2] GeForce GTX TITAN 2 | 71'C, 100 %,  ?? %,  250 /  ?? W |     0 / 12189 MB | (Not Supported)
"""  # noqa: E501


MB = 1024 * 1024


def remove_ansi_codes(s):
    import re
    s = re.compile(r'\x1b[^m]*m').sub('', s)
    s = re.compile(r'\x0f').sub('', s)
    return s


class TestGPUStat(unittest.TestCase):

    @mock.patch('psutil.Process')
    @mock.patch('gpustat.core.N')
    def test_main(self, N, Process):
        """
        Test whether gpustat.main() works well. The behavior is mocked
        exactly as in test_new_query_mocked().
        """
        _configure_mock(N, Process)
        sys.argv = ['gpustat']
        gpustat.main()

    @mock.patch('psutil.Process')
    @mock.patch('gpustat.core.N')
    def test_new_query_mocked(self, N, Process):
        """
        A basic functionality test, in a case where everything is just normal.
        """
        _configure_mock(N, Process)

        gpustats = gpustat.new_query()
        fp = StringIO()
        gpustats.print_formatted(
            fp=fp, no_color=False, show_user=True,
            show_cmd=True, show_pid=True, show_power=True, show_fan_speed=True
        )

        result = fp.getvalue()
        print(result)

        unescaped = remove_ansi_codes(result)
        # remove first line (header)
        unescaped = '\n'.join(unescaped.split('\n')[1:])

        self.maxDiff = 4096
        self.assertEqual(unescaped, MOCK_EXPECTED_OUTPUT_FULL)

    @mock.patch('psutil.Process')
    @mock.patch('gpustat.core.N')
    def test_new_query_mocked_nonexistent_pid(self, N, Process):
        """
        Test a case where nvidia query returns non-existent pids (see #16, #18)
        """
        _configure_mock(N, Process, scenario_nonexistent_pid=True)

        gpustats = gpustat.new_query()
        gpustats.print_formatted(fp=sys.stdout)

    @mock.patch('psutil.Process')
    @mock.patch('gpustat.core.N')
    def test_attributes_and_items(self, N, Process):
        """
        Test whether each property of `GPUStat` instance is well-defined.
        """
        _configure_mock(N, Process)

        g = gpustat.new_query()[1]  # includes N/A
        print("(keys) : %s" % str(g.keys()))
        print(g)

        self.assertEqual(g['name'], g.entry['name'])
        self.assertEqual(g['uuid'], g.uuid)

        with self.assertRaises(KeyError):
            g['unknown_key']

        print("uuid : %s" % g.uuid)
        print("name : %s" % g.name)
        print("memory : used %d total %d avail %d" % (
            g.memory_used, g.memory_total, g.memory_available))
        print("temperature : %d" % (g.temperature))
        print("utilization : %s" % (g.utilization))

    @unittest.skipIf(sys.version_info < (3, 4), "Only in Python 3.4+")
    @mock.patch('psutil.Process')
    @mock.patch('gpustat.core.N')
    def test_args_endtoend(self, N, Process):
        """
        End-to-end testing given command line args.
        """
        _configure_mock(N, Process)

        def capture_output(*args):
            f = StringIO()
            import contextlib

            with contextlib.redirect_stdout(f):  # requires python 3.4+
                try:
                    gpustat.main(*args)
                except SystemExit:
                    raise AssertionError(
                        "Argparse failed (see above error message)"
                    )
            return f.getvalue()

        s = capture_output('gpustat', )
        unescaped = remove_ansi_codes(s)
        # remove first line (header)
        unescaped = '\n'.join(unescaped.split('\n')[1:])
        self.maxDiff = 4096
        self.assertEqual(unescaped, MOCK_EXPECTED_OUTPUT_DEFAULT)

        s = capture_output('gpustat', '--no-header')
        self.assertIn("[0]", s.split('\n')[0])

    @mock.patch('psutil.Process')
    @mock.patch('gpustat.core.N')
    def test_json_mocked(self, N, Process):
        _configure_mock(N, Process)
        gpustats = gpustat.new_query()

        fp = StringIO()
        gpustats.print_json(fp=fp)

        import json
        j = json.loads(fp.getvalue())
        print(j)


if __name__ == '__main__':
    unittest.main()
