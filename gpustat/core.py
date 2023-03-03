#!/usr/bin/env python

"""
Implementation of gpustat

@author Jongwook Choi
@url https://github.com/wookayin/gpustat
"""

from typing import Sequence
import json
import locale
import os.path
import platform
import sys
import time
from datetime import datetime
from io import StringIO

import psutil
from blessed import Terminal

import gpustat.util as util
from gpustat.nvml import pynvml as N


NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024

DEFAULT_GPUNAME_WIDTH = 16

IS_WINDOWS = 'windows' in platform.platform().lower()


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
    def available(self):
        return True

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
    def utilization_enc(self):
        """
        Returns the GPU encoder utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.enc']
        return int(v) if v is not None else None

    @property
    def utilization_dec(self):
        """
        Returns the GPU decoder utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.dec']
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

    def print_to(self, fp, *,
                 with_colors=True,    # deprecated arg
                 show_cmd=False,
                 show_full_cmd=False,
                 no_processes=False,
                 show_user=False,
                 show_pid=False,
                 show_fan_speed=None,
                 show_codec="",
                 show_power=None,
                 gpuname_width=None,
                 eol_char=os.linesep,
                 term=None,
                 ):
        if term is None:
            term = Terminal(stream=sys.stdout)

        # color settings
        colors = {}

        def _conditional(cond_fn, true_value, false_value,
                         error_value=term.bold_black):
            try:
                return cond_fn() and true_value or false_value
            except Exception:
                return error_value

        _ENC_THRESHOLD = 50

        colors['C0'] = term.normal
        colors['C1'] = term.cyan
        colors['CBold'] = term.bold
        colors['CName'] = _conditional(lambda: self.available,
                                       term.blue, term.red)
        colors['CTemp'] = _conditional(lambda: self.temperature < 50,
                                       term.red, term.bold_red)
        colors['FSpeed'] = _conditional(lambda: self.fan_speed < 30,
                                        term.cyan, term.bold_cyan)
        colors['CMemU'] = _conditional(lambda: self.available,
                                       term.bold_yellow, term.bold_black)
        colors['CMemT'] = _conditional(lambda: self.available,
                                       term.yellow, term.bold_black)
        colors['CMemP'] = term.yellow
        colors['CCPUMemU'] = term.yellow
        colors['CUser'] = term.bold_black   # gray
        colors['CUtil'] = _conditional(lambda: self.utilization < 30,
                                       term.green, term.bold_green)
        colors['CUtilEnc'] = _conditional(
            lambda: self.utilization_enc < _ENC_THRESHOLD,
            term.green, term.bold_green)
        colors['CUtilDec'] = _conditional(
            lambda: self.utilization_dec < _ENC_THRESHOLD,
            term.green, term.bold_green)
        colors['CCPUUtil'] = term.green
        colors['CPowU'] = _conditional(
            lambda: (self.power_limit is not None and
                     float(self.power_draw) / self.power_limit < 0.4),
            term.magenta, term.bold_magenta
        )
        colors['CPowL'] = term.magenta
        colors['CCmd'] = term.color(24)   # a bit dark

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value='??'):
            return none_value if v is None else v

        # build one-line display information
        # we want power use optional, but if deserves being grouped with
        # temperature and utilization
        reps = u"%(C1)s[{entry[index]}]%(C0)s "
        if gpuname_width is None or gpuname_width != 0:
            reps += u"%(CName)s{entry_name:{gpuname_width}}%(C0)s |"
        reps += u"%(CTemp)s{entry[temperature.gpu]:>3}°C%(C0)s, "

        if show_fan_speed:
            reps += "%(FSpeed)s{entry[fan.speed]:>3} %%%(C0)s, "

        reps += "%(CUtil)s{entry[utilization.gpu]:>3} %%%(C0)s"

        if show_codec:
            codec_info = []
            if "enc" in show_codec:
                codec_info.append(
                    "%(CBold)sE: %(C0)s"
                    "%(CUtilEnc)s{entry[utilization.enc]:>3} %%%(C0)s")
            if "dec" in show_codec:
                codec_info.append(
                    "%(CBold)sD: %(C0)s"
                    "%(CUtilDec)s{entry[utilization.dec]:>3} %%%(C0)s")
            reps += " ({})".format("  ".join(codec_info))

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

        class entry_repr_accessor:
            def __init__(self, entry):
                self.entry = entry
            def __getitem__(self, key):
                return _repr(self.entry[key])

        reps = reps.format(
            entry=entry_repr_accessor(self.entry),
            entry_name=util.shorten_left(
                self.entry["name"], width=gpuname_width, placeholder='…'),
            gpuname_width=gpuname_width or DEFAULT_GPUNAME_WIDTH
        )

        # Add " |" only if processes information is to be added.
        if not no_processes:
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

        def full_process_info(p):
            r = "{C0} ├─ {:>6} ".format(
                    _repr(p['pid'], '--'), **colors
                )
            r += "{C0}({CCPUUtil}{:4.0f}%{C0}, {CCPUMemU}{:>6}{C0})".format(
                    _repr(p['cpu_percent'], '--'),
                    util.bytes2human(_repr(p['cpu_memory_usage'], 0)), **colors
                )
            full_command_pretty = util.prettify_commandline(
                p['full_command'], colors['C1'], colors['CCmd'])
            r += "{C0}: {CCmd}{}{C0}".format(
                _repr(full_command_pretty, '?'),
                **colors
            )
            return r

        processes = self.entry['processes']
        full_processes = []
        if processes is None and not no_processes:
            # None (not available)
            reps += ' ({})'.format(NOT_SUPPORTED)
        elif not no_processes:
            for p in processes:
                reps += ' ' + process_repr(p)
                if show_full_cmd:
                    full_processes.append(eol_char + full_process_info(p))
        if show_full_cmd and full_processes:
            full_processes[-1] = full_processes[-1].replace('├', '└', 1)
            reps += ''.join(full_processes)
        fp.write(reps)
        return fp

    def jsonify(self):
        o = self.entry.copy()
        if self.entry['processes'] is not None:
            o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                              for p in self.entry['processes']]
        return o


class InvalidGPU(GPUStat):
    class FallbackDict(dict):
        def __missing__(self, key):
            return "?"

    def __init__(self, gpu_index, message, ex):
        super().__init__(self.FallbackDict(
            index=gpu_index,
            name=message,
            processes=None
        ))
        self.exception = ex

    @property
    def available(self):
        return False


class GPUStatCollection(Sequence[GPUStat]):

    global_processes = {}

    def __init__(self, gpu_list, driver_version=None):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()
        self.driver_version = driver_version

    @staticmethod
    def clean_processes():
        for pid in list(GPUStatCollection.global_processes.keys()):
            if not psutil.pid_exists(pid):
                del GPUStatCollection.global_processes[pid]

    @staticmethod
    def new_query(debug=False, id=None) -> 'GPUStatCollection':
        """Query the information of all the GPUs on local machine"""

        N.nvmlInit()
        log = util.DebugHelper()

        def _decode(b):
            if isinstance(b, bytes):
                return b.decode('utf-8')    # for python3, to unicode
            return b

        def get_gpu_info(handle):
            """Get one GPU information specified by nvml handle"""

            def safepcall(fn, error_value):
                # Ignore the exception from psutil when the process is gone
                # at the moment of querying. See #144.
                return util.safecall(
                    fn, error_value=error_value,
                    exc_types=(psutil.AccessDenied, psutil.NoSuchProcess,
                               FileNotFoundError))

            def get_process_info(nv_process):
                """Get the process information of specific pid"""
                process = {}
                if nv_process.pid not in GPUStatCollection.global_processes:
                    GPUStatCollection.global_processes[nv_process.pid] = \
                        psutil.Process(pid=nv_process.pid)
                ps_process: psutil.Process = GPUStatCollection.global_processes[nv_process.pid]

                # TODO: ps_process is being cached, but the dict below is not.
                process['username'] = safepcall(ps_process.username, '?')
                # cmdline returns full path;
                # as in `ps -o comm`, get short cmdnames.
                _cmdline = safepcall(ps_process.cmdline, [])
                if not _cmdline:
                    # sometimes, zombie or unknown (e.g. [kworker/8:2H])
                    process['command'] = '?'
                    process['full_command'] = ['?']
                else:
                    process['command'] = os.path.basename(_cmdline[0])
                    process['full_command'] = _cmdline
                # Bytes to MBytes
                # if drivers are not TTC this will be None.
                usedmem = nv_process.usedGpuMemory // MB if \
                          nv_process.usedGpuMemory else None
                process['gpu_memory_usage'] = usedmem

                process['cpu_percent'] = safepcall(ps_process.cpu_percent, 0.0)
                process['cpu_memory_usage'] = safepcall(
                    lambda: round((ps_process.memory_percent() / 100.0) *
                                  psutil.virtual_memory().total),
                    0.0)

                process['pid'] = nv_process.pid
                return process

            name = _decode(N.nvmlDeviceGetName(handle))
            uuid = _decode(N.nvmlDeviceGetUUID(handle))

            try:
                temperature = N.nvmlDeviceGetTemperature(
                    handle, N.NVML_TEMPERATURE_GPU
                )
            except N.NVMLError as e:
                log.add_exception("temperature", e)
                temperature = None  # Not supported

            try:
                fan_speed = N.nvmlDeviceGetFanSpeed(handle)
            except N.NVMLError as e:
                log.add_exception("fan_speed", e)
                fan_speed = None  # Not supported

            try:
                # memory: in Bytes
                # Note that this is a compat-patched API (see gpustat.nvml)
                memory = N.nvmlDeviceGetMemoryInfo(handle)
            except N.NVMLError as e:
                log.add_exception("memory", e)
                memory = None  # Not supported

            try:
                utilization = N.nvmlDeviceGetUtilizationRates(handle)
            except N.NVMLError as e:
                log.add_exception("utilization", e)
                utilization = None  # Not supported

            try:
                utilization_enc = N.nvmlDeviceGetEncoderUtilization(handle)
            except N.NVMLError as e:
                log.add_exception("utilization_enc", e)
                utilization_enc = None  # Not supported

            try:
                utilization_dec = N.nvmlDeviceGetDecoderUtilization(handle)
            except N.NVMLError as e:
                log.add_exception("utilization_dec", e)
                utilization_dec = None  # Not supported

            try:
                power = N.nvmlDeviceGetPowerUsage(handle)
            except N.NVMLError as e:
                log.add_exception("power", e)
                power = None

            try:
                power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
            except N.NVMLError as e:
                log.add_exception("power_limit", e)
                power_limit = None

            try:
                nv_comp_processes = \
                    N.nvmlDeviceGetComputeRunningProcesses(handle)
            except N.NVMLError as e:
                log.add_exception("compute_processes", e)
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = \
                    N.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except N.NVMLError as e:
                log.add_exception("graphics_processes", e)
                nv_graphics_processes = None  # Not supported

            if nv_comp_processes is None and nv_graphics_processes is None:
                processes = None
            else:
                processes = []
                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                # A single process might run in both of graphics and compute mode,
                # However we will display the process only once
                seen_pids = set()
                for nv_process in nv_comp_processes + nv_graphics_processes:
                    if nv_process.pid in seen_pids:
                        continue
                    seen_pids.add(nv_process.pid)
                    try:
                        process = get_process_info(nv_process)
                        processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass
                    except psutil.AccessDenied:
                        pass
                    except FileNotFoundError:
                        # Ignore the exception which probably has occured
                        # from psutil, due to a non-existent PID (see #95).
                        # The exception should have been translated, but
                        # there appears to be a bug of psutil. It is unlikely
                        # FileNotFoundError is thrown in different situations.
                        pass

                # TODO: Do not block if full process info is not requested
                time.sleep(0.1)
                for process in processes:
                    pid = process['pid']
                    cache_process: psutil.Process = GPUStatCollection.global_processes[pid]
                    process['cpu_percent'] = safepcall(cache_process.cpu_percent, 0)

            index = N.nvmlDeviceGetIndex(handle)

            # GPU Info.
            # We use the same key/spec as per `nvidia-smi --query-help-gpu`
            gpu_info = {
                'index': index,
                'uuid': uuid,
                'name': name,
                'temperature.gpu': temperature,
                'fan.speed': fan_speed,
                'utilization.gpu': utilization.gpu if utilization else None,
                'utilization.enc':
                    utilization_enc[0] if utilization_enc else None,
                'utilization.dec':
                    utilization_dec[0] if utilization_dec else None,
                'power.draw': power // 1000 if power is not None else None,
                'enforced.power.limit': power_limit // 1000
                if power_limit is not None else None,
                # Convert bytes into MBytes
                'memory.used': memory.used // MB if memory else None,
                'memory.total': memory.total // MB if memory else None,
                'processes': processes,
            }
            GPUStatCollection.clean_processes()
            return gpu_info

        # 1. get the list of gpu and status
        gpu_list = []
        device_count = N.nvmlDeviceGetCount()

        if id is None:
            gpus_to_query = range(device_count)
        elif isinstance(id, str):
            gpus_to_query = [int(i) for i in id.split(',')]
        elif isinstance(id, Sequence):
            gpus_to_query = [int(i) for i in id]
        else:
            raise TypeError(f"Unknown id: {id}")

        for index in gpus_to_query:
            try:
                handle = N.nvmlDeviceGetHandleByIndex(index)
                gpu_info = get_gpu_info(handle)
                gpu_stat = GPUStat(gpu_info)
            except N.NVMLError_Unknown as e:
                gpu_stat = InvalidGPU(index, "((Unknown Error))", e)
            except N.NVMLError_GpuIsLost as e:
                gpu_stat = InvalidGPU(index, "((GPU is lost))", e)

            if isinstance(gpu_stat, InvalidGPU):
                log.add_exception("GPU %d" % index, gpu_stat.exception)
            gpu_list.append(gpu_stat)

        # 2. additional info (driver version, etc).
        try:
            driver_version = _decode(N.nvmlSystemGetDriverVersion())
        except N.NVMLError as e:
            log.add_exception("driver_version", e)
            driver_version = None    # N/A

        if debug:
            log.report_summary()

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

    def print_formatted(self, fp=sys.stdout, *,
                        force_color=False, no_color=False,
                        show_cmd=False, show_full_cmd=False, show_user=False,
                        show_pid=False, show_fan_speed=None,
                        show_codec="", show_power=None,
                        gpuname_width=None, show_header=True,
                        no_processes=False,
                        eol_char=os.linesep,
                        ):
        # ANSI color configuration
        if force_color and no_color:
            raise ValueError("--color and --no_color can't"
                             " be used at the same time")

        if force_color:
            TERM = os.getenv('TERM') or 'xterm-256color'
            t_color = Terminal(kind=TERM, force_styling=True)

            # workaround of issue #32 (watch doesn't recognize sgr0 characters)
            t_color._normal = u'\x1b[0;10m'
        elif no_color:
            t_color = Terminal(force_styling=None)
        else:
            t_color = Terminal()   # auto, depending on isatty

        # appearance settings
        if gpuname_width is None:
            gpuname_width = max([len(g.entry['name']) for g in self] + [0])

        # header
        if show_header:
            if IS_WINDOWS:
                # no localization is available; just use a reasonable default
                # same as str(timestr) but without ms
                timestr = self.query_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_format = locale.nl_langinfo(locale.D_T_FMT)
                timestr = self.query_time.strftime(time_format)
            header_template = '{t.bold_white}{hostname:{width}}{t.normal}  '
            header_template += '{timestr}  '
            header_template += '{t.bold_black}{driver_version}{t.normal}'

            header_msg = header_template.format(
                    hostname=self.hostname,
                    width=(gpuname_width or DEFAULT_GPUNAME_WIDTH) + 3,  # len("[?]")
                    timestr=timestr,
                    driver_version=self.driver_version,
                    t=t_color,
                )

            fp.write(header_msg.strip())
            fp.write(eol_char)

        # body
        for g in self:
            g.print_to(fp,
                       show_cmd=show_cmd,
                       show_full_cmd=show_full_cmd,
                       no_processes=no_processes,
                       show_user=show_user,
                       show_pid=show_pid,
                       show_fan_speed=show_fan_speed,
                       show_codec=show_codec,
                       show_power=show_power,
                       gpuname_width=gpuname_width,
                       eol_char=eol_char,
                       term=t_color)
            fp.write(eol_char)

        if len(self.gpus) == 0:
            print(t_color.yellow("(No GPUs are available)"))

        fp.flush()

    def jsonify(self):
        return {
            'hostname': self.hostname,
            'driver_version': self.driver_version,
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
        fp.write(os.linesep)
        fp.flush()


def new_query() -> GPUStatCollection:
    '''
    Obtain a new GPUStatCollection instance by querying nvidia-smi
    to get the list of GPUs and running process information.
    '''
    return GPUStatCollection.new_query()


def gpu_count() -> int:
    '''Return the number of available GPUs in the system.'''
    try:
        N.nvmlInit()
        return N.nvmlDeviceGetCount()
    except N.NVMLError:
        return 0  # fallback
    finally:
        try:
            N.nvmlShutdown()
        except N.NVMLError:
            pass


def is_available() -> bool:
    '''Return True if the NVML library and GPU devices are available.'''
    return gpu_count() > 0
