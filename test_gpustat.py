"""
Unit or integration tests for gpustat
"""

from __future__ import print_function

import unittest
import gpustat

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

# mock output for test
def _mock_check_output(cmd, shell=True):
    if cmd.startswith('nvidia-smi --query-compute-apps'):
        return u'''\
GPU-10fb0fbd-2696-43f3-467f-d280d906a107, 48448, 4000
GPU-10fb0fbd-2696-43f3-467f-d280d906a107, 153223, 4000
GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2, 192453, 3000
GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2, 194826, 6000
GPU-50205d95-57b6-f541-2bcb-86c09afed564, 38310, 4245
GPU-50205d95-57b6-f541-2bcb-86c09afed564, [Not Supported], [Not Supported]
'''
    elif cmd.startswith('nvidia-smi --query-gpu'):
        return u'''\
0, GPU-10fb0fbd-2696-43f3-467f-d280d906a107, GeForce GTX TITAN X, 80, 76, 8000, 12287
1, GPU-d1df4664-bb44-189c-7ad0-ab86c8cb30e2, GeForce GTX TITAN X, 36, 0, 9000, 12287
2, GPU-50205d95-57b6-f541-2bcb-86c09afed564, GeForce GTX TITAN X, 71, [Not Supported], 8520, 12287
'''
    elif cmd.startswith('ps -o pid,user:16,comm -p'):
        return u'''\
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


def remove_ansi_codes(s):
    import re
    ansi_escape = re.compile(r'\x1b[^m]*m')
    return ansi_escape.sub('', s)

class TestGPUStat(unittest.TestCase):

    def test_new_query_mocked(self):
        gpustats = gpustat.new_query()
        fp = StringIO()
        gpustats.print_formatted(fp=fp, no_color=False, show_user=True, show_cmd=True, show_pid=True)

        result = fp.getvalue()
        print(result)

        unescaped = remove_ansi_codes(result)
        self.assertEqual(unescaped,
"""\
[0] GeForce GTX TITAN X | 80'C,  76 % |  8000 / 12287 MB | user1:python/48448(4000M) user2:python/153223(4000M)
[1] GeForce GTX TITAN X | 36'C,   0 % |  9000 / 12287 MB | user1:torch/192453(3000M) user3:caffe/194826(6000M)
[2] GeForce GTX TITAN X | 71'C,  ?? % |  8520 / 12287 MB | user3:python/38310(4245M) --:--/--(?M)
""")
        #"""

    def test_attributes_and_items(self):
        g = gpustat.new_query()[0]

        self.assertEqual(g['name'], g.entry['name'])
        self.assertEqual(g['uuid'], g.uuid)

        with self.assertRaises(KeyError):
            g['unknown_key']



if __name__ == '__main__':
    unittest.main()
