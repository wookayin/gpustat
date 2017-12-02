#!/usr/bin/env python

"""
the gpustat script :)

@author Jongwook Choi
@url https://github.com/wookayin/gpustat
"""

from __future__ import print_function
from datetime import datetime
from collections import defaultdict
from six.moves import cStringIO as StringIO

import six
import sys
import locale
import platform
import json
import psutil
import os.path

import pynvml as N
from blessings import Terminal

__version__ = '0.4.1'


NOT_SUPPORTED = 'Not Supported'


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError('entry should be a dict, {} given'.format(type(entry)))
        self.entry = entry

        # Handle '[Not Supported] for old GPU cards (#6)
        for k in self.entry.keys():
            if isinstance(self.entry[k], six.string_types) and NOT_SUPPORTED in self.entry[k]:
                self.entry[k] = None


    def __repr__(self):
        return self.print_to(StringIO()).getvalue()

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    @property
    def index(self):
        """
        Returns the index of GPU (as in nvidia-smi).
        """
        return self.entry['index']

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
    def memory_free(self):
        """
        Returns the free (available) memory (in MB) as an integer.
        """
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def memory_available(self):
        """
        Returns the available memory (in MB) as an integer. Alias of memory_free.
        """
        return self.memory_free

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

    @property
    def power_draw(self):
        """
        Returns the GPU power usage in Watts,
        or None if the information is not available.
        """
        v = self.entry['power.draw']
        return int(v) if v is not None else None

    @property
    def power_limit(self):
        """
        Returns the (enforced) GPU power limit in Watts,
        or None if the information is not available.
        """
        v = self.entry['enforced.power.limit']
        return int(v) if v is not None else None

    @property
    def processes(self):
        """
        Get the list of running processes on the GPU.
        """
        return list(self.entry['processes'])


    def print_to(self, fp,
                 with_colors=True,    # deprecated arg
                 show_cmd=False,
                 show_user=False,
                 show_pid=False,
                 show_power=None,
                 gpuname_width=16,
                 term=Terminal(),
                 ):
        # color settings
        colors = {}
        def _conditional(cond_fn, true_value, false_value,
                         error_value=term.bold_black):
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
        colors['CUser'] = term.bold_black   # gray
        colors['CUtil'] = _conditional(lambda: int(self.entry['utilization.gpu']) < 30,
                                       term.green, term.bold_green)
        colors['CPowU'] = _conditional(lambda: float(self.entry['power.draw']) / self.entry['enforced.power.limit'] < 0.4,
                                       term.magenta, term.bold_magenta)
        colors['CPowL'] = term.magenta

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value='??'):
            if v is None: return none_value
            else: return str(v)

        # build one-line display information
        # we want power use optional, but if deserves being grouped with temperature and utilization
        reps = "%(C1)s[{entry[index]}]%(C0)s %(CName)s{entry[name]:{gpuname_width}}%(C0)s |" \
               "%(CTemp)s{entry[temperature.gpu]:>3}'C%(C0)s, %(CUtil)s{entry[utilization.gpu]:>3} %%%(C0)s"

        if show_power:
            reps += ",  %(CPowU)s{entry[power.draw]:>3}%(C0)s "
            if show_power == True or 'limit' in show_power:
                reps += "/ %(CPowL)s{entry[enforced.power.limit]:>3}%(C0)s "
                reps += "%(CPowL)sW%(C0)s"
            else:
                reps += "%(CPowU)sW%(C0)s"

        reps += " | %(C1)s%(CMemU)s{entry[memory.used]:>5}%(C0)s / %(CMemT)s{entry[memory.total]:>5}%(C0)s MB"
        reps = (reps) % colors
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
        else:
            # None (not available)
            reps += ' (Not Supported)'


        fp.write(reps)
        return fp

    def jsonify(self):
        o = dict(self.entry)
        o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                          for p in self.entry['processes']]
        return o


class GPUStatCollection(object):

    def __init__(self, gpu_list):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()

    @staticmethod
    def new_query():
        """Query the information of all the GPUs on local machine"""

        N.nvmlInit()

        def get_gpu_info(handle):
            """Get one GPU information specified by nvml handle"""

            def get_process_info(pid):
                """Get the process information of specific pid"""
                process = {}
                ps_process = psutil.Process(pid=pid)
                process['username'] = ps_process.username()
                # cmdline returns full path; as in `ps -o comm`, get short cmdnames.
                _cmdline = ps_process.cmdline()
                if not _cmdline:   # sometimes, zombie or unknown (e.g. [kworker/8:2H])
                    process['command'] = '?'
                else:
                    process['command'] = os.path.basename(_cmdline[0])
                # Bytes to MBytes
                process['gpu_memory_usage'] = int(nv_process.usedGpuMemory / 1024 / 1024)
                process['pid'] = nv_process.pid
                return process

            def _decode(b):
                if isinstance(b, bytes):
                    return b.decode()    # for python3, to unicode
                return b

            name = _decode(N.nvmlDeviceGetName(handle))
            uuid = _decode(N.nvmlDeviceGetUUID(handle))

            try:
                temperature = N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)
            except N.NVMLError:
                temperature = None  # Not supported

            try:
                memory = N.nvmlDeviceGetMemoryInfo(handle) # in Bytes
            except N.NVMLError:
                memory = None  # Not supported

            try:
                utilization = N.nvmlDeviceGetUtilizationRates(handle)
            except N.NVMLError:
                utilization = None  # Not supported

            try:
                power = N.nvmlDeviceGetPowerUsage(handle)
            except:
                power = None

            try:
                power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
            except:
                power_limit = None

            processes = []
            try:
                nv_comp_processes = N.nvmlDeviceGetComputeRunningProcesses(handle)
            except N.NVMLError:
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = N.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except N.NVMLError:
                nv_graphics_processes = None  # Not supported

            if nv_comp_processes is None and nv_graphics_processes is None:
                processes = None   # Not supported (in both cases)
            else:
                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                for nv_process in (nv_comp_processes + nv_graphics_processes):
                    # TODO: could be more information such as system memory usage,
                    # CPU percentage, create time etc.
                    try:
                        process = get_process_info(nv_process.pid)
                        processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass

            gpu_info = {
                'index': index,
                'uuid': uuid,
                'name': name,
                'temperature.gpu': temperature,
                'utilization.gpu': utilization.gpu if utilization else None,
                'power.draw': int(power / 1000) if power is not None else None,
                'enforced.power.limit': int(power_limit / 1000) if power_limit is not None else None,
                # Convert bytes into MBytes
                'memory.used': int(memory.used / 1024 / 1024) if memory else None,
                'memory.total': int(memory.total / 1024 / 1024) if memory else None,
                'processes': processes,
            }
            return gpu_info

        # 1. get the list of gpu and status
        gpu_list = []
        device_count = N.nvmlDeviceGetCount()

        for index in range(device_count):
            handle = N.nvmlDeviceGetHandleByIndex(index)
            gpu_info = get_gpu_info(handle)
            gpu_stat = GPUStat(gpu_info)
            gpu_list.append(gpu_stat)

        N.nvmlShutdown()
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

    def print_formatted(self, fp=sys.stdout, force_color=False, no_color=False,
                        show_cmd=False, show_user=False, show_pid=False,
                        show_power=None, gpuname_width=16,
                        ):
        # ANSI color configuration
        if force_color and no_color:
            raise ValueError("--color and --no_color can't be used at the same time")

        if force_color:
            t_color = Terminal(kind='linux', force_styling=True)
        elif no_color:
            t_color = Terminal(force_styling=None)
        else:
            t_color = Terminal()   # auto, depending on isatty

        # header
        time_format = locale.nl_langinfo(locale.D_T_FMT)

        header_msg = '{t.bold_white}{hostname}{t.normal}  {timestr}'.format(**{
            'hostname' : self.hostname,
            'timestr' : self.query_time.strftime(time_format),
            't' : t_color,
        })

        fp.write(header_msg)
        fp.write('\n')

        # body
        gpuname_width = max([gpuname_width] + [len(g.entry['name']) for g in self])
        for g in self:
            g.print_to(fp,
                       show_cmd=show_cmd,
                       show_user=show_user,
                       show_pid=show_pid,
                       show_power=show_power,
                       gpuname_width=gpuname_width,
                       term=t_color)
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
                raise TypeError(type(obj))

        o = self.jsonify()
        json.dump(o, fp, indent=4, separators=(',', ': '),
                  default=date_handler)
        fp.write('\n')
        fp.flush()


def new_query():
    '''
    Obtain a new GPUStatCollection instance by querying nvidia-smi
    to get the list of GPUs and running process information.
    '''
    return GPUStatCollection.new_query()


def print_gpustat(json=False, debug=False, **args):
    '''
    Display the GPU query results into standard output.
    '''
    try:
        gpu_stats = GPUStatCollection.new_query()
    except Exception as e:
        sys.stderr.write('Error on querying NVIDIA devices. Use --debug flag for details\n')
        if debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if json:
        gpu_stats.print_json(sys.stdout)
    else:
        gpu_stats.print_formatted(sys.stdout, **args)


def main():
    # attach SIGPIPE handler to properly handle broken pipe
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # arguments to gpustat
    import argparse
    parser = argparse.ArgumentParser()

    parser_color = parser.add_mutually_exclusive_group()
    parser_color.add_argument('--force-color', '--color', action='store_true',
                              help='Force to output with colors')
    parser_color.add_argument('--no-color', action='store_true',
                              help='Suppress colored output')

    parser.add_argument('-c', '--show-cmd', action='store_true',
                        help='Display cmd name of running process')
    parser.add_argument('-u', '--show-user', action='store_true',
                        help='Display username of running process')
    parser.add_argument('-p', '--show-pid', action='store_true',
                        help='Display PID of running process')
    parser.add_argument('-P', '--show-power', nargs='?', const='draw,limit',
                        choices=['', 'draw', 'limit', 'draw,limit', 'limit,draw'],
                        help='Show GPU power usage or draw (and/or limit)')
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
