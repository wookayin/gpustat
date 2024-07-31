"""
Implementation of gpustat

@author Jongwook Choi
@url https://github.com/wookayin/gpustat
"""

import functools
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Sequence, Union, cast)

try:
    from typing_extensions import TypedDict
except ModuleNotFoundError:
    TypedDict = None
# pyright: reportOptionalOperand = false
# pyright: reportTypedDictNotRequiredAccess = false
# pylint: disable=redefined-builtin

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

from gpustat import util

if util.hasAMD():
    from gpustat import rocml as nvml
    from gpustat import rocml as N
    from gpustat.rocml import check_driver_nvml_version
else:
    from gpustat import nvml
    from gpustat.nvml import pynvml as N
    from gpustat.nvml import check_driver_nvml_version

NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024

DEFAULT_GPUNAME_WIDTH = 16

IS_WINDOWS = 'windows' in platform.platform().lower()


# Types
NVMLHandle = Any  # N.c_nvmlDevice_t
Megabytes = int
Celcius = int
Percentage = int
Watts = int
ProcessInfo = Dict[str, Any]

# We use the same key/spec as `nvidia-smi --query-help-gpu`
NvidiaGPUInfo = TypedDict('NvidiaGPUInfo', {
    'index': int,
    'name': str,
    'uuid': str,
    'temperature.gpu': Optional[Celcius],
    'fan.speed': Optional[Percentage],
    'utilization.gpu': Optional[Percentage],
    'utilization.enc': Optional[Percentage],
    'utilization.dec': Optional[Percentage],
    'power.draw': Optional[Watts],
    'enforced.power.limit': Optional[Watts],
    'memory.used': Megabytes,
    'memory.total': Megabytes,
    'processes': Optional[List[ProcessInfo]],
}, total=False) if TYPE_CHECKING else dict  # type: ignore


class GPUStat:

    def __init__(self, entry: NvidiaGPUInfo):
        if not isinstance(entry, dict):
            raise TypeError(
                'entry should be a dict, {} given'.format(type(entry))
            )
        self.entry = entry

    def __repr__(self) -> str:
        return self.print_to(StringIO()).getvalue()

    def keys(self) -> Iterable[str]:
        return self.entry.keys()

    def __getitem__(self, key) -> Any:
        return self.entry[key]

    @property
    def available(self) -> bool:
        return True

    @property
    def index(self) -> int:
        """Returns the index of GPU (as in nvidia-smi --query-gpu=index)."""
        return self.entry['index']

    @property
    def uuid(self) -> str:
        """Returns the uuid of GPU (as nvidia-smi --query-gpu=uuid).

        e.g. GPU-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry['uuid']

    @property
    def name(self) -> str:
        """Returns the name of GPU card (e.g. GeForce Titan X)."""
        return self.entry['name']

    @property
    def memory_total(self) -> Megabytes:
        """Returns the total memory (in MB) as an integer."""
        return int(self.entry['memory.total'])

    @property
    def memory_used(self) -> Megabytes:
        """Returns the occupied memory (in MB) as an integer."""
        return int(self.entry['memory.used'])

    @property
    def memory_free(self) -> Megabytes:
        """Returns the free (available) memory (in MB) as an integer."""
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def memory_available(self) -> Megabytes:
        """Returns the available memory (in MB) as an integer.

        Alias to memory_free.
        """
        return self.memory_free

    @property
    def temperature(self) -> Optional[Celcius]:
        """
        Returns the temperature (in Celcius) of GPU as an integer,
        or None if the information is not available.
        """
        v = self.entry['temperature.gpu']
        return int(v) if v is not None else None

    @property
    def fan_speed(self) -> Optional[Percentage]:
        """
        Returns the fan speed percentage (0-100) of maximum intended speed
        as an integer, or None if the information is not available.
        """
        v = self.entry['fan.speed']
        return int(v) if v is not None else None

    @property
    def utilization(self) -> Optional[Percentage]:
        """
        Returns the GPU utilization (in percentage),
        or None if the information is not available.
        """
        v = self.entry['utilization.gpu']
        return int(v) if v is not None else None

    @property
    def utilization_enc(self) -> Optional[Percentage]:
        """
        Returns the GPU encoder utilization (in percentage),
        or None if the information is not available.
        """
        v = self.entry['utilization.enc']
        return int(v) if v is not None else None

    @property
    def utilization_dec(self) -> Optional[Percentage]:
        """
        Returns the GPU decoder utilization (in percentage),
        or None if the information is not available.
        """
        v = self.entry['utilization.dec']
        return int(v) if v is not None else None

    @property
    def power_draw(self) -> Optional[Percentage]:
        """
        Returns the GPU power usage in Watts,
        or None if the information is not available.
        """
        v = self.entry['power.draw']
        return int(v) if v is not None else None

    @property
    def power_limit(self) -> Optional[Watts]:
        """
        Returns the (enforced) GPU power limit in Watts,
        or None if the information is not available.
        """
        v = self.entry['enforced.power.limit']
        return int(v) if v is not None else None

    @property
    def processes(self) -> Optional[List[ProcessInfo]]:
        """Get the list of running processes on the GPU."""
        return self.entry['processes']

    @property
    def clk_freq(self) -> Optional[int]:
        """
        """
        v = self.entry['clk_freq']
        return int(v) if v is not None else None

    @property
    def clk_freq_max(self) -> Optional[int]:
        """
        """
        v = self.entry['clk_freq_max']
        return int(v) if v is not None else None

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
            except Exception:  # pylint: disable=broad-exception-caught
                return error_value

        _ENC_THRESHOLD = 50

        colors['C0'] = term.normal
        colors['C1'] = term.cyan
        colors['CName'] = _conditional(lambda: self.available, term.blue, term.red)
        colors['CTemp'] = _conditional(lambda: self.temperature < 50, term.red, term.bold_red)
        colors['FSpeed'] = _conditional(lambda: self.fan_speed < 30, term.cyan, term.bold_cyan)
        colors['CMemU'] = _conditional(lambda: self.available, term.bold_yellow, term.bold_black)
        colors['CMemT'] = _conditional(lambda: self.available, term.yellow, term.bold_black)
        colors['CMemP'] = term.yellow
        colors['CCPUMemU'] = term.yellow
        colors['CUser'] = term.bold_black   # gray
        colors['CUtil'] = _conditional(lambda: self.utilization < 30, term.green, term.bold_green)
        colors['CUtilEnc'] = _conditional(
            lambda: self.utilization_enc < _ENC_THRESHOLD,
            term.green, term.bold_green)
        colors['CUtilDec'] = _conditional(
            lambda: self.utilization_dec < _ENC_THRESHOLD,
            term.green, term.bold_green)
        colors['CCPUUtil'] = term.green
        colors['CPowU'] = _conditional(
            lambda: (self.power_limit is not None and
                     float(self.power_draw) / self.power_limit < 0.4),  # type: ignore
            term.magenta, term.bold_magenta
        )
        colors['CPowL'] = term.magenta
        colors['CCmd'] = term.color(24)   # a bit dark

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value: Any = '??'):
            return none_value if v is None else v

        # build one-line display information
        # we want power use optional, but if deserves being grouped with
        # temperature and utilization

        reps = []
        def _write(*args, color=None, end=''):
            args = [str(x) for x in args]
            if color:
                # pylint: disable-next=consider-using-get
                if color in colors:
                    color = colors[color]
                args = [color] + args + [term.normal]
            if end:
                args.append(end)
            reps.extend(args)

        def rjustify(x, size):
            return f"{x:>{size}}"

        # Null-safe property accessor like self.xxxx,
        # but fall backs to '?' for None or missing values
        class SafePropertyAccessor:
            def __init__(self, obj):
                self.obj = obj
            def __getattr__(self, name):  # type: ignore
                try:
                    v = getattr(self.obj, name)
                    return v if v is not None else '??'
                except TypeError:  # possibly int(None), etc.
                    return '??'

        safe_self = cast(GPUStat, SafePropertyAccessor(self))

        _write(f"[{self.index}]", color=term.cyan)
        _write(" ")

        if gpuname_width is None or gpuname_width != 0:
            gpuname_width = gpuname_width or DEFAULT_GPUNAME_WIDTH
            _write(f"{util.shorten_left(self.name, width=gpuname_width, placeholder='…'):{gpuname_width}}",
                   color='CName')
            _write(" |")

        _write(rjustify(safe_self.temperature, 3), "°C", color='CTemp', end=', ')

        if show_fan_speed:
            _write(rjustify(safe_self.fan_speed, 3), " %", color='FSpeed', end=', ')

        _write(rjustify(safe_self.utilization, 3), " %", color='CUtil')

        if show_codec:
            _write(" (")
            _sep = ''
            if "enc" in show_codec:
                _write("E: ", color=term.bold)
                _write(rjustify(safe_self.utilization_enc, 3), " %", color='CUtilEnc')
                _sep = '  '  # TODO comma?
            if "dec" in show_codec:
                _write(_sep, "D: ", color=term.bold)
                _write(rjustify(safe_self.utilization_dec, 3), " %", color='CUtilDec')
            _write(")")

        if show_power:
            _write(",  ")
            _write(rjustify(safe_self.power_draw, 3), color='CPowU')
            if show_power is True or 'limit' in show_power:
                _write(" / ")
                _write(rjustify(safe_self.power_limit, 3), ' W', color='CPowL')
                
        _write(",  ")
        _write(rjustify(safe_self.clk_freq, 3), color='CPowU')
        _write(" / ")
        _write(rjustify(safe_self.clk_freq_max, 3), color='CPowU')
        _write(" MHz")

        # Memory
        _write(" | ")
        _write(rjustify(safe_self.memory_used, 5), color='CMemU')
        _write(" / ")
        _write(rjustify(safe_self.memory_total, 5), color='CMemT')
        _write(" MB")

        # Add " |" only if processes information is to be added.
        if not no_processes:
            _write(" |")

        def process_repr(p: ProcessInfo):
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

        def full_process_info(p: ProcessInfo):
            r = "{C0} ├─ {:>6} ".format(
                    _repr(p['pid'], '--'), **colors
                )
            r += "{C0}({CCPUUtil}{:4.0f}%{C0}, {CCPUMemU}{:>6}{C0})".format(
                    _repr(p['cpu_percent'], '--'),
                    util.bytes2human(_repr(p['cpu_memory_usage'], 0)), **colors
                )  # type: ignore
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
            _write(' ', '(', NOT_SUPPORTED, ')')
        elif not no_processes:
            for p in (processes or []):
                _write(' ', process_repr(p))
                if show_full_cmd:
                    full_processes.append(eol_char + full_process_info(p))
        if show_full_cmd and full_processes:
            full_processes[-1] = full_processes[-1].replace('├', '└', 1)
            _write(''.join(full_processes))

        fp.write(''.join(reps))
        return fp

    def jsonify(self):
        o = self.entry.copy()
        if self.entry['processes'] is not None:
            o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                              for p in self.entry['processes']]
        return o


class InvalidGPU(GPUStat):
    class FallbackDict(dict):
        # pylint: disable-next=useless-return
        def __missing__(self, key):
            del key
            return None

    def __init__(self, gpu_index, message, ex):
        super().__init__(self.FallbackDict(
            index=gpu_index,
            name=message,
            processes=None,
        ))  # type: ignore
        self.exception = ex

    @property
    def available(self):
        return False


class GPUStatCollection(Sequence[GPUStat]):

    global_processes = {}

    def __init__(self,
                 gpu_list: Sequence[GPUStat],
                 driver_version: Optional[str] = None):
        self.gpus = list(gpu_list)

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

        nvml.ensure_initialized()
        log = util.DebugHelper()

        def _decode(b: Union[str, bytes]) -> str:
            if isinstance(b, bytes):
                return b.decode('utf-8')
            assert isinstance(b, str)
            return b

        def get_gpu_info(handle: NVMLHandle, index: int = None) -> NvidiaGPUInfo:
            """Get one GPU information specified by nvml handle"""

            def safepcall(fn: Callable[[], Any], error_value: Any):
                # Ignore the exception from psutil when the process is gone
                # at the moment of querying. See #144.
                return util.safecall(
                    fn, error_value=error_value,
                    exc_types=(psutil.AccessDenied, psutil.NoSuchProcess,
                               FileNotFoundError))

            def get_process_info(nv_process) -> ProcessInfo:
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

            def safenvml(fn):
                @functools.wraps(fn)
                def _wrapped(*args, **kwargs):
                    try:
                        return fn(*args, **kwargs)
                    except N.NVMLError as e:
                        log.add_exception(fn.__name__, e)
                        return None  # Not supported
                return _wrapped

            gpu_info = NvidiaGPUInfo()
            gpu_info['index'] = N.nvmlDeviceGetIndex(handle) if index is None else index

            gpu_info['name'] = _decode(N.nvmlDeviceGetName(handle))
            gpu_info['uuid'] = _decode(N.nvmlDeviceGetUUID(handle))

            gpu_info['temperature.gpu'] = safenvml(
                N.nvmlDeviceGetTemperature)(handle, N.NVML_TEMPERATURE_GPU)

            gpu_info['fan.speed'] = safenvml(N.nvmlDeviceGetFanSpeed)(handle)

            # memory: in Bytes
            # Note that this is a compat-patched API (see gpustat.nvml)
            memory = N.nvmlDeviceGetMemoryInfo(handle)
            gpu_info['memory.used'] = int(memory.used) // MB
            gpu_info['memory.total'] = int(memory.total) // MB

            # GPU utilization
            utilization = safenvml(N.nvmlDeviceGetUtilizationRates)(handle)
            gpu_info['utilization.gpu'] = int(utilization.gpu) if utilization is not None else None

            utilization = safenvml(N.nvmlDeviceGetEncoderUtilization)(handle)
            gpu_info['utilization.enc'] = utilization[0] if utilization is not None else None

            utilization = safenvml(N.nvmlDeviceGetDecoderUtilization)(handle)
            gpu_info['utilization.dec'] = utilization[0] if utilization is not None else None

            # Power
            power = safenvml(N.nvmlDeviceGetPowerUsage)(handle)
            gpu_info['power.draw'] = power if power is not None else None

            power_limit = safenvml(N.nvmlDeviceGetEnforcedPowerLimit)(handle)
            gpu_info['enforced.power.limit'] = power_limit if power_limit is not None else None

            # Frequency
            freq = safenvml(N.nvmlDeviceGetClkFreq)(handle)
            gpu_info['clk_freq'] = freq if freq is not None else None
            freq_max = safenvml(N.nvmlDeviceGetClkFreqMax)(handle)
            gpu_info['clk_freq_max'] = freq_max if freq_max is not None else None

            # Processes
            nv_comp_processes = safenvml(N.nvmlDeviceGetComputeRunningProcesses)(handle)
            nv_graphics_processes = safenvml(N.nvmlDeviceGetGraphicsRunningProcesses)(handle)

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
            gpu_info['processes'] = processes

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
                handle: NVMLHandle = N.nvmlDeviceGetHandleByIndex(index)
                gpu_info = get_gpu_info(handle, index)
                gpu_stat = GPUStat(gpu_info)
            except N.NVMLError_Unknown as e:
                gpu_stat = InvalidGPU(index, "((Unknown Error))", e)
            except N.NVMLError_GpuIsLost as e:
                gpu_stat = InvalidGPU(index, "((GPU is lost))", e)
            except Exception as e:
                gpu_stat = InvalidGPU(index, "((Unknown Error))", e)

            if isinstance(gpu_stat, InvalidGPU):
                log.add_exception("GPU %d" % index, gpu_stat.exception)
            gpu_list.append(gpu_stat)

        # 2. additional info (driver version, etc).
        # TODO: check this only once, no need to call multiple times
        try:
            driver_version = _decode(N.nvmlSystemGetDriverVersion())
            check_driver_nvml_version(driver_version)
        except N.NVMLError as e:
            log.add_exception("driver_version", e)
            driver_version = None    # N/A

        if debug:
            log.report_summary()

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
            # pylint: disable-next=protected-access
            t_color._normal = '\x1b[0;10m'  # type: ignore
        elif no_color:
            t_color = Terminal(force_styling=None)  # type: ignore
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
        nvml.ensure_initialized()
        return N.nvmlDeviceGetCount()
    except N.NVMLError:
        return 0  # fallback


def is_available() -> bool:
    '''Return True if the NVML library and GPU devices are available.'''
    return gpu_count() > 0
