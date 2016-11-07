"""
Unit or integration tests for gpustat
"""

import unittest
import gpustat

# mock output for test
def _mock_check_output(cmd, shell=True):
    if cmd.startswith('nvidia-smi --query-compute-apps'):
        return '''\
GPU-10fb0fbd-2696-43f3-467f-d280d906a107, 48448, 4000
GPU-10fb0fbd-2696-43f3-467f-d280d906a107, 153223, 4000
GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2, 192453, 3000
GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2, 194826, 6000
GPU-50205d95-57b6-f541-2bcb-86c09afed564, 38310, 4245
GPU-50205d95-57b6-f541-2bcb-86c09afed564, [Not Supported], [Not Supported]
'''
    elif cmd.startswith('nvidia-smi --query-gpu'):
        return '''\
0, GPU-10fb0fbd-2696-43f3-467f-d280d906a107, GeForce GTX TITAN X, 80, 76, 8000, 12287
1, GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2, GeForce GTX TITAN X, 36, 0, 9000, 12287
2, GPU-50205d95-57b6-f541-2bcb-86c09afed564, GeForce GTX TITAN X, 71, [Not Supported], 8520, 12287
'''
    elif cmd.startswith('ps -o pid,user,comm -p'):
        return '''\
   PID USER  COMMAND
 48448 user1 python
154213 user1 caffe
 38310 user3 python
153223 user2 python
194826 user3 caffe
192453 user1 torch
'''
    else:
        raise ValueError(cmd)

# mocking (override subprocess.check_output)
gpustat.check_output = _mock_check_output


class TestGPUStat(unittest.TestCase):

    def test_new_query_mocked(self):
        gpustats = gpustat.new_query()
        gpustats.print_formatted(no_color=False, show_user=True, show_cmd=True, show_pid=True)


if __name__ == '__main__':
    unittest.main()
