#!/usr/bin/env python

"""
the gpustat script :)

@author Jongwook Choi
@url https://github.com/wookayin/gpustat
"""

from __future__ import print_function
from subprocess import check_output, CalledProcessError
from datetime import datetime
from collections import defaultdict
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import sys
import locale
import platform
import json

import psutil
# wildcard import because the name is too long
from pynvml import *
from blessings import Terminal

__version__ = '0.4.0.dev'


NOT_SUPPPORTED = 'Not Supported'

term = Terminal()

def execute_process(command_shell):
    stdout = check_output(command_shell, shell=True).strip()
    if not isinstance(stdout, (str)):
        stdout = stdout.decode()
    return stdout


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError('entry should be a dict, {} given'.format(type(entry)))
        self.entry = entry

        # Handle '[Not Supported] for old GPU cards (#6)
        for k in self.entry.keys():
            if self.entry[k] == NOT_SUPPPORTED:
                self.entry[k] = None


    def __repr__(self):
        return self.print_to(StringIO()).getvalue()

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    @property
    def uuid(self):
        """
        Returns the uuid returned by nvidia-smi,
        e.g. GPU-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry['uuid']

    @property
    def name(self):
        """
        Returns the name of GPU card (e.g. Geforce Titan X)
        """
        return self.entry['name']

    @property
    def memory_total(self):
        """
        Returns the total memory (in MB) as an integer.
        """
        return int(self.entry['memory.total'])

    @property
    def memory_used(self):
        """
        Returns the occupied memory (in MB) as an integer.
        """
        return int(self.entry['memory.used'])

    @property
    def memory_available(self):
        """
        Returns the available memory (in MB) as an integer.
        """
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def temperature(self):
        """
        Returns the temperature of GPU as an integer,
        or None if the information is not available.
        """
        v = self.entry['temperature.gpu']
        return int(v) if v is not None else None

    @property
    def utilization(self):
        """
        Returns the GPU utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.gpu']
        return int(v) if v is not None else None

    def print_to(self, fp,
                 with_colors=True,
                 show_cmd=False,
                 show_user=False,
                 show_pid=False,
                 gpuname_width=16
                 ):
        # color settings
        colors = {}
        def _conditional(cond_fn, true_value, false_value,
                         error_value=term.gray):
            try:
                if cond_fn(): return true_value
                else: return false_value
            except:
                return error_value

        colors['C0'] = term.normal
        colors['C1'] = term.cyan
        colors['CName'] = term.blue
        colors['CTemp'] = _conditional(lambda: int(self.entry['temperature.gpu']) < 50,
                                       term.red, term.bold_red)
        colors['CMemU'] = term.bold_yellow
        colors['CMemT'] = term.yellow
        colors['CMemP'] = term.yellow
        colors['CUser'] = term.gray
        colors['CUtil'] = _conditional(lambda: int(self.entry['utilization.gpu']) < 30,
                                       term.green, term.bold_green)

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value='??'):
            if v is None: return none_value
            else: return str(v)

        # build one-line display information
        reps = ("%(C1)s[{entry[index]}]%(C0)s %(CName)s{entry[name]:{gpuname_width}}%(C0)s |" +
                "%(CTemp)s{entry[temperature.gpu]:>3}'C%(C0)s, %(CUtil)s{entry[utilization.gpu]:>3} %%%(C0)s | " +
                "%(C1)s%(CMemU)s{entry[memory.used]:>5}%(C0)s / %(CMemT)s{entry[memory.total]:>5}%(C0)s MB"
                ) % colors
        reps = reps.format(entry={k: _repr(v) for (k, v) in self.entry.items()},
                           gpuname_width=gpuname_width)
        reps += " |"

        def process_repr(p):
            r = ''
            if not show_cmd or show_user:
                r += "{CUser}{}{C0}".format(_repr(p['username'], '--'), **colors)
            if show_cmd:
                if r: r += ':'
                r += "{C1}{}{C0}".format(_repr(p.get('command', p['pid']), '--'), **colors)

            if show_pid:
                r += ("/%s" % _repr(p['pid'], '--'))
            r += '({CMemP}{}M{C0})'.format(_repr(p['gpu_memory_usage'], '?'), **colors)
            return r

        if self.entry['processes'] is not None:
            for p in self.entry['processes']:
                reps += ' ' + process_repr(p)

        fp.write(reps)
        return fp

    @property
    def uuid(self):
        return self.entry['uuid']

    def jsonify(self):
        o = dict(self.entry)
        o['processes'] = [{k: v for (k, v) in p.iteritems() if k != 'gpu_uuid'}
                          for p in self.processes]
        return o

    def add_process(self, p):
        self.processes.append(p)
        return self


class GPUStatCollection(object):

    def __init__(self, gpu_list):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()

    @staticmethod
    def new_query():
        """Query the information of all the GPUs on local machine"""

        def get_gpu_info(handle):
            """Get one GPU information specified by nvml handle"""

            def get_process_info(pid):
                """Get the process information of specific pid"""
                process = {}
                ps_process = psutil.Process(pid=pid)
                process['username'] = ps_process.username()
                process['command'] = ps_process.cmdline()[0]
                # Bytes to MBytes
                process['gpu_memory_usage'] = nv_process.usedGpuMemory / 1024 / 1024
                process['pid'] = nv_process.pid
                return process

            name = nvmlDeviceGetName(handle)
            uuid = nvmlDeviceGetUUID(handle)
            temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            memory = nvmlDeviceGetMemoryInfo(handle) # in Bytes

            util_supported = True
            try:
                utilization = nvmlDeviceGetUtilizationRates(handle)
            except NVMLError:
                util_supported = False

            processes = []
            try:
                nv_processes = nvmlDeviceGetComputeRunningProcesses(handle)
                # dict type is mutable
                for nv_process in nv_processes:
                    #TODO: could be more information such as system memory usage,
                    # CPU percentage, create time etc.
                    process = get_process_info(nv_process.pid)
                    processes.append(process)
            except NVMLError:
                processes = NOT_SUPPPORTED

            gpu_info={
                'index': index,
                'uuid': uuid,
                'name': name,
                'temperature.gpu': temperature,
                'utilization.gpu': utilization.gpu if util_supported else NOT_SUPPPORTED,
                'memory.used': memory.used / 1024 / 1024, # Convert bytes into MBytes
                'memory.total': memory.total / 1024 / 1024,
                'processes': processes,
            }
            return gpu_info

        nvmlInit()
        # 1. get the list of gpu and status
        gpu_list = []
        device_count = nvmlDeviceGetCount()

        for index in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(index)
            gpu_info = get_gpu_info(handle)
            gpu_stat = GPUStat(gpu_info)
            gpu_list.append(gpu_stat)

        nvmlShutdown()
        return GPUStatCollection(gpu_list)

    def __len__(self):
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, index):
        return self.gpus[index]

    def __repr__(self):
        s = 'GPUStatCollection(host=%s, [\n' % self.hostname
        s += '\n'.join('  ' + str(g) for g in self.gpus)
        s += '\n])'
        return s

    # --- Printing Functions ---

    def print_formatted(self, fp=sys.stdout, no_color=False,
                        show_cmd=False, show_user=False, show_pid=False,
                        gpuname_width=16,
                        ):
        # header
        time_format = locale.nl_langinfo(locale.D_T_FMT)
        header_msg = '{t.white}{hostname}{t.normal}  {timestr}'.format(**{
            'hostname' : self.hostname,
            'timestr' : self.query_time.strftime(time_format),
            't': term,
        })

        print(header_msg)

        # body
        gpuname_width = max([gpuname_width] + [len(g.entry['name']) for g in self])
        for g in self:
            g.print_to(fp,
                       with_colors=not no_color,
                       show_cmd=show_cmd,
                       show_user=show_user,
                       show_pid=show_pid,
                       gpuname_width=gpuname_width)
            fp.write('\n')

        fp.flush()

    def jsonify(self):
        return {
            'hostname' : self.hostname,
            'query_time' : self.query_time,
            "gpus" : [g.jsonify() for g in self]
        }

    def print_json(self, fp=sys.stdout):
        def date_handler(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                raise TypeError

        o = self.jsonify()
        json.dump(o, fp, indent=4, separators=(',', ': '),
                  default=date_handler)
        fp.write('\n')
        fp.flush()


def print_gpustat(json=False, debug=False, **args):
    '''
    Display the GPU query results into standard output.
    '''
    try:
        gpu_stats = GPUStatCollection.new_query()
    except CalledProcessError:
        sys.stderr.write('Error on calling nvidia-smi. Use --debug flag for details\n')
        if debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if json:
        gpu_stats.print_json(sys.stdout)
    else:
        gpu_stats.print_formatted(sys.stdout, **args)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-color', action='store_true',
                        help='Suppress colored output')
    parser.add_argument('-c', '--show-cmd', action='store_true',
                        help='Display cmd name of running process')
    parser.add_argument('-u', '--show-user', action='store_true',
                        help='Display username of running process')
    parser.add_argument('-p', '--show-pid', action='store_true',
                        help='Display PID of running process')
    parser.add_argument('--gpuname-width', type=int, default=16,
                        help='The minimum column width of GPU names, defaults to 16')
    parser.add_argument('--json', action='store_true', default=False,
                        help='Print all the information in JSON format')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Allow to print additional informations for debugging.')
    parser.add_argument('-v', '--version', action='version',
                        version=('gpustat %s' % __version__))
    args = parser.parse_args()

    print_gpustat(**vars(args))

if __name__ == '__main__':
    main()
