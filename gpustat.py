#!/usr/bin/env python

"""
the gpustat script :)

@author Jongwook Choi
@url https://github.com/wookayin/gpustat
"""

from __future__ import print_function
from subprocess import check_output, CalledProcessError
from datetime import datetime
from collections import OrderedDict, defaultdict
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import sys
import locale
import platform
import json

__version__ = '0.3.1'


class ANSIColors:
    RESET   = '\033[0m'
    WHITE   = '\033[1m'
    RED     = '\033[0;31m'
    GREEN   = '\033[0;32m'
    YELLOW  = '\033[0;33m'
    BLUE    = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN    = '\033[0;36m'
    GRAY        = '\033[1;30m'
    BOLD_RED    = '\033[1;31m'
    BOLD_GREEN  = '\033[1;32m'
    BOLD_YELLOW = '\033[1;33m'

    @staticmethod
    def wrap(color, msg):
        return (color + msg + ANSIColors.RESET)


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
        self.processes = []

        # Handle '[Not Supported] for old GPU cards (#6)
        for k in self.entry.keys():
            if 'Not Supported' in self.entry[k]:
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
                         error_value=ANSIColors.GRAY):
            try:
                if cond_fn(): return true_value
                else: return false_value
            except:
                return error_value

        colors['C0'] = ANSIColors.RESET
        colors['C1'] = ANSIColors.CYAN
        colors['CName'] = ANSIColors.BLUE
        colors['CTemp'] = _conditional(lambda: int(self.entry['temperature.gpu']) < 50,
                                       ANSIColors.RED, ANSIColors.BOLD_RED)
        colors['CMemU'] = ANSIColors.BOLD_YELLOW
        colors['CMemT'] = ANSIColors.YELLOW
        colors['CMemP'] = ANSIColors.YELLOW
        colors['CUser'] = ANSIColors.GRAY
        colors['CUtil'] = _conditional(lambda: int(self.entry['utilization.gpu']) < 30,
                                       ANSIColors.GREEN, ANSIColors.BOLD_GREEN)

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
                r += "{CUser}{}{C0}".format(_repr(p['user'], '--'), **colors)
            if show_cmd:
                if r: r += ':'
                r += "{C1}{}{C0}".format(_repr(p.get('comm', p['pid']), '--'), **colors)

            if show_pid:
                r += ("/%s" % _repr(p['pid'], '--'))
            r += '({CMemP}{}M{C0})'.format(_repr(p['used_memory'], '?'), **colors)
            return r

        for p in self.processes:
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
        self.gpus = OrderedDict()
        for g in gpu_list:
            self.gpus[g.uuid] = g

        # attach process information (owner, pid, etc.)
        self.update_process_information()

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()

    @staticmethod
    def new_query():
        # 1. get the list of gpu and status
        gpu_query_columns = ('index', 'uuid', 'name', 'temperature.gpu',
                             'utilization.gpu', 'memory.used', 'memory.total')
        gpu_list = []

        smi_output = execute_process(
            r'nvidia-smi --query-gpu={query_cols} --format=csv,noheader,nounits'.format(
                query_cols=','.join(gpu_query_columns)
            ))

        for line in smi_output.split('\n'):
            if not line: continue
            query_results = line.split(',')

            g = GPUStat({col_name: col_value.strip() for
                         (col_name, col_value) in zip(gpu_query_columns, query_results)
                         })
            gpu_list.append(g)

        return GPUStatCollection(gpu_list)

    @staticmethod
    def running_processes():
        # 1. collect all running GPU processes
        gpu_query_columns = ('gpu_uuid', 'pid', 'used_memory')
        smi_output = execute_process(
            r'nvidia-smi --query-compute-apps={query_cols} --format=csv,noheader,nounits'.format(
                query_cols=','.join(gpu_query_columns)
            ))

        process_entries = []
        for line in smi_output.split('\n'):
            if not line: continue
            query_results = line.split(',')
            process_entry = dict({col_name: col_value.strip() for
                                  (col_name, col_value) in zip(gpu_query_columns, query_results)
                                  })
            process_entries.append(process_entry)

        pid_map = {int(e['pid']) : None for e in process_entries
                   if not 'Not Supported' in e['pid']}

        # 2. map pid to username, etc.
        if pid_map:
            pid_output = execute_process('ps -o {} -p {}'.format(
                'pid,user:16,comm',
                ','.join(map(str, pid_map.keys()))
            ))
            for line in pid_output.split('\n'):
                if (not line) or 'PID' in line: continue
                pid, user, comm = line.split()[:3]
                pid_map[int(pid)] = {
                    'user' : user,
                    'comm' : comm
                }

        # 3. add some process information to each process_entry
        for process_entry in process_entries[:]:

            if 'Not Supported' in process_entry['pid']:
                # TODO move this stuff into somewhere appropriate
                # such as running_processes(): process_entry = ...
                # or introduce Process class to elegantly handle it
                process_entry['user'] = None
                process_entry['comm'] = None
                process_entry['pid'] = None
                process_entry['used_memory'] = None
                continue

            pid = int(process_entry['pid'])

            if pid_map[pid] is None:
                # !?!? this pid is listed up in nvidia-smi's query result,
                # but actually seems not to be a valid running process. ignore!
                process_entries.remove(process_entry)
                continue

            process_entry.update(pid_map[pid])

        return process_entries

    def update_process_information(self):
        processes = self.running_processes()
        for p in processes:
            try:
                g = self.gpus[p['gpu_uuid']]
            except KeyError:
                # ignore?
                pass
            g.add_process(p)
        return self

    def __len__(self):
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus.values())

    def __getitem__(self, index):
        return list(self.gpus.values())[index]

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
        header_msg = '%(WHITE)s{hostname}%(RESET)s  {timestr}'.format(**{
            'hostname' : self.hostname,
            'timestr' : self.query_time.strftime(time_format)

        }) % (defaultdict(str) if no_color else ANSIColors.__dict__)

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


def self_test():
    gpu_stats = GPUStatCollection.new_query()
    print('# of GPUS:', len(gpu_stats))
    for g in gpu_stats:
        print(g)

    process_entries = GPUStatCollection.running_processes()
    print('---Entries---')
    print(process_entries)

    print('-------------')


def new_query():
    '''
    Obtain a new GPUStatCollection instance by querying nvidia-smi
    to get the list of GPUs and running process information.
    '''
    return GPUStatCollection.new_query()


def print_gpustat(json=False, **args):
    '''
    Display the GPU query results into standard output.
    '''
    try:
        gpu_stats = GPUStatCollection.new_query()
    except CalledProcessError:
        sys.stderr.write('Error on calling nvidia-smi\n')
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
    parser.add_argument('-v', '--version', action='version',
                        version=('gpustat %s' % __version__))
    args = parser.parse_args()

    print_gpustat(**vars(args))

if __name__ == '__main__':
    main()
