#!/usr/bin/env python

"""
Implementation of gpustat

@author Jongwook Choi
@url https://github.com/wookayin/gpustat
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import locale
import os.path
import platform
import sys
from datetime import datetime

from six.moves import cStringIO as StringIO

import psutil
import pynvml as N
from blessings import Terminal

NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError(
                'entry should be a dict, {} given'.format(type(entry))
            )
        self.entry = entry

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
        Returns the available memory (in MB) as an integer.
        Alias of memory_free.
        """
        return self.memory_free

    @property
    def temperature(self):
        """
        Returns the temperature (in celcius) of GPU as an integer,
        or None if the information is not available.
        """
        v = self.entry['temperature.gpu']
        return int(v) if v is not None else None

    @property
    def fan_speed(self):
        """
        Returns the fan speed percentage (0-100) of maximum intended speed
        as an integer, or None if the information is not available.
        """
        v = self.entry['fan.speed']
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
        return self.entry['processes']

    def print_to(self, fp,
                 with_colors=True,    # deprecated arg
                 show_cmd=False,
                 show_user=False,
                 show_pid=False,
                 show_power=None,
                 show_fan_speed=None,
                 gpuname_width=16,
                 term=Terminal(),
                 ):
        # color settings
        colors = {}

        def _conditional(cond_fn, true_value, false_value,
                         error_value=term.bold_black):
            try:
                return cond_fn() and true_value or false_value
            except Exception:
                return error_value

        colors['C0'] = term.normal
        colors['C1'] = term.cyan
        colors['CName'] = term.blue
        colors['CTemp'] = _conditional(lambda: self.temperature < 50,
                                       term.red, term.bold_red)
        colors['FSpeed'] = _conditional(lambda: self.fan_speed < 30,
                                        term.cyan, term.bold_cyan)
        colors['CMemU'] = term.bold_yellow
        colors['CMemT'] = term.yellow
        colors['CMemP'] = term.yellow
        colors['CUser'] = term.bold_black   # gray
        colors['CUtil'] = _conditional(lambda: self.utilization < 30,
                                       term.green, term.bold_green)
        colors['CPowU'] = _conditional(
            lambda: float(self.power_draw) / self.power_limit < 0.4,
            term.magenta, term.bold_magenta
        )
        colors['CPowL'] = term.magenta

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value='??'):
            return none_value if v is None else v

        # build one-line display information
        # we want power use optional, but if deserves being grouped with
        # temperature and utilization
        reps = "%(C1)s[{entry[index]}]%(C0)s " \
            "%(CName)s{entry[name]:{gpuname_width}}%(C0)s |" \
            "%(CTemp)s{entry[temperature.gpu]:>3}'C%(C0)s, "

        if show_fan_speed:
            reps += "%(FSpeed)s{entry[fan.speed]:>3} %%%(C0)s, "

        reps += "%(CUtil)s{entry[utilization.gpu]:>3} %%%(C0)s"

        if show_power:
            reps += ",  %(CPowU)s{entry[power.draw]:>3}%(C0)s "
            if show_power is True or 'limit' in show_power:
                reps += "/ %(CPowL)s{entry[enforced.power.limit]:>3}%(C0)s "
                reps += "%(CPowL)sW%(C0)s"
            else:
                reps += "%(CPowU)sW%(C0)s"

        reps += " | %(C1)s%(CMemU)s{entry[memory.used]:>5}%(C0)s " \
            "/ %(CMemT)s{entry[memory.total]:>5}%(C0)s MB"
        reps = (reps) % colors
        reps = reps.format(entry={k: _repr(v) for k, v in self.entry.items()},
                           gpuname_width=gpuname_width)
        reps += " |"

        def process_repr(p):
            r = ''
            if not show_cmd or show_user:
                r += "{CUser}{}{C0}".format(
                    _repr(p['username'], '--'), **colors
                )
            if show_cmd:
                if r:
                    r += ':'
                r += "{C1}{}{C0}".format(
                    _repr(p.get('command', p['pid']), '--'), **colors
                )

            if show_pid:
                r += ("/%s" % _repr(p['pid'], '--'))
            r += '({CMemP}{}M{C0})'.format(
                _repr(p['gpu_memory_usage'], '?'), **colors
            )
            return r

        processes = self.entry['processes']
        if processes is None:
            # None (not available)
            reps += ' ({})'.format(NOT_SUPPORTED)
        else:
            for p in processes:
                reps += ' ' + process_repr(p)

        fp.write(reps)
        return fp

    def jsonify(self):
        o = dict(self.entry)
        if self.entry['processes'] is not None:
            o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                              for p in self.entry['processes']]
        else:
            o['processes'] = '({})'.format(NOT_SUPPORTED)
        return o


class GPUStatCollection(object):

    def __init__(self, gpu_list, driver_version=None):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()
        self.driver_version = driver_version

    @staticmethod
    def new_query():
        """Query the information of all the GPUs on local machine"""

        N.nvmlInit()

        def _decode(b):
            if isinstance(b, bytes):
                return b.decode()    # for python3, to unicode
            return b

        def get_gpu_info(handle):
            """Get one GPU information specified by nvml handle"""

            def get_process_info(nv_process):
                """Get the process information of specific pid"""
                process = {}
                ps_process = psutil.Process(pid=nv_process.pid)
                process['username'] = ps_process.username()
                # cmdline returns full path;
                # as in `ps -o comm`, get short cmdnames.
                _cmdline = ps_process.cmdline()
                if not _cmdline:
                    # sometimes, zombie or unknown (e.g. [kworker/8:2H])
                    process['command'] = '?'
                else:
                    process['command'] = os.path.basename(_cmdline[0])
                # Bytes to MBytes
                process['gpu_memory_usage'] = nv_process.usedGpuMemory // MB
                process['pid'] = nv_process.pid
                return process

            name = _decode(N.nvmlDeviceGetName(handle))
            uuid = _decode(N.nvmlDeviceGetUUID(handle))

            try:
                temperature = N.nvmlDeviceGetTemperature(
                    handle, N.NVML_TEMPERATURE_GPU
                )
            except N.NVMLError:
                temperature = None  # Not supported

            try:
                fan_speed = N.nvmlDeviceGetFanSpeed(handle)
            except N.NVMLError:
                fan_speed = None  # Not supported

            try:
                memory = N.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
            except N.NVMLError:
                memory = None  # Not supported

            try:
                utilization = N.nvmlDeviceGetUtilizationRates(handle)
            except N.NVMLError:
                utilization = None  # Not supported

            try:
                power = N.nvmlDeviceGetPowerUsage(handle)
            except N.NVMLError:
                power = None

            try:
                power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
            except N.NVMLError:
                power_limit = None

            try:
                nv_comp_processes = \
                    N.nvmlDeviceGetComputeRunningProcesses(handle)
            except N.NVMLError:
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = \
                    N.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except N.NVMLError:
                nv_graphics_processes = None  # Not supported

            if nv_comp_processes is None and nv_graphics_processes is None:
                processes = None
            else:
                processes = []
                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                for nv_process in nv_comp_processes + nv_graphics_processes:
                    # TODO: could be more information such as system memory
                    # usage, CPU percentage, create time etc.
                    try:
                        process = get_process_info(nv_process)
                        processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass

            index = N.nvmlDeviceGetIndex(handle)
            gpu_info = {
                'index': index,
                'uuid': uuid,
                'name': name,
                'temperature.gpu': temperature,
                'fan.speed': fan_speed,
                'utilization.gpu': utilization.gpu if utilization else None,
                'power.draw': power // 1000 if power is not None else None,
                'enforced.power.limit': power_limit // 1000
                if power_limit is not None else None,
                # Convert bytes into MBytes
                'memory.used': memory.used // MB if memory else None,
                'memory.total': memory.total // MB if memory else None,
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

        # 2. additional info (driver version, etc).
        try:
            driver_version = _decode(N.nvmlSystemGetDriverVersion())
        except N.NVMLError:
            driver_version = None    # N/A

        N.nvmlShutdown()
        return GPUStatCollection(gpu_list, driver_version=driver_version)

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
                        show_power=None, show_fan_speed=None, gpuname_width=16,
                        show_header=True,
                        eol_char=os.linesep,
                        ):
        # ANSI color configuration
        if force_color and no_color:
            raise ValueError("--color and --no_color can't"
                             " be used at the same time")

        if force_color:
            t_color = Terminal(kind='linux', force_styling=True)

            # workaround of issue #32 (watch doesn't recognize sgr0 characters)
            t_color.normal = u'\x1b[0;10m'
        elif no_color:
            t_color = Terminal(force_styling=None)
        else:
            t_color = Terminal()   # auto, depending on isatty

        # appearance settings
        entry_name_width = [len(g.entry['name']) for g in self]
        gpuname_width = max([gpuname_width or 0] + entry_name_width)

        # header
        if show_header:
            time_format = locale.nl_langinfo(locale.D_T_FMT)

            header_template = '{t.bold_white}{hostname:{width}}{t.normal}  '
            header_template += '{timestr}  '
            header_template += '{t.bold_black}{driver_version}{t.normal}'

            header_msg = header_template.format(
                    hostname=self.hostname,
                    width=gpuname_width + 3,  # len("[?]")
                    timestr=self.query_time.strftime(time_format),
                    driver_version=self.driver_version,
                    t=t_color,
                )

            fp.write(header_msg.strip())
            fp.write(eol_char)

        # body
        for g in self:
            g.print_to(fp,
                       show_cmd=show_cmd,
                       show_user=show_user,
                       show_pid=show_pid,
                       show_power=show_power,
                       show_fan_speed=show_fan_speed,
                       gpuname_width=gpuname_width,
                       term=t_color)
            fp.write(eol_char)

        fp.flush()

    def jsonify(self):
        return {
            'hostname': self.hostname,
            'query_time': self.query_time,
            "gpus": [g.jsonify() for g in self]
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
