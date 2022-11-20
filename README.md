`gpustat`
=========

[![pypi](https://img.shields.io/pypi/v/gpustat.svg?maxAge=86400)][pypi_gpustat]
[![Build Status](https://travis-ci.org/wookayin/gpustat.svg?branch=master)](https://travis-ci.org/wookayin/gpustat)
[![license](https://img.shields.io/github/license/wookayin/gpustat.svg?maxAge=86400)](LICENSE)

Just *less* than nvidia-smi?

![Screenshot: gpustat -cp](https://github.com/wookayin/gpustat/blob/master/screenshot.png)

NOTE: This works with NVIDIA Graphics Devices only, no AMD support as of now. Contributions are welcome!

Self-Promotion: A web interface of `gpustat` is available (in alpha)! Check out [gpustat-web][gpustat-web].

[gpustat-web]: https://github.com/wookayin/gpustat-web



Quick Installation
------------------

Install from [PyPI][pypi_gpustat]:

```
pip install gpustat
```

If you don't have root (sudo) privilege, please try installing `gpustat` on user namespace: `pip install --user gpustat`.

To install the latest version (master branch) via pip:

```
pip install git+https://github.com/wookayin/gpustat.git@master
```


### NVIDIA Driver Requirements

`gpustat` uses [NVIDIA's official python bindings for NVML library (pynvml)][pypi_pynvml]. As of now `gpustat` requires `nvidia-ml-py >= 11.450.129, <=11.495.46`, which is compatible with NVIDIA driver versions R450.00 or higher. Please upgrade the NVIDIA driver if `gpustat` fails to display process information. If your NVIDIA driver is too old, you can use older `gpustat` versions (`pip install gpustat<1.0`). See [#107][gh-issue-107] for more details.


### Python requirements

- gpustat<1.0: Compatible with python 2.7 and >=3.4
- gpustat 1.0: [Python >= 3.4][gh-issue-66]
- gpustat 1.1: Python >= 3.6


Usage
-----

`$ gpustat`

Options (Please see `gpustat --help` for more details):

* `--color`            : Force colored output (even when stdout is not a tty)
* `--no-color`         : Suppress colored output
* `-u`, `--show-user`  : Display username of the process owner
* `-c`, `--show-cmd`   : Display the process name
* `-f`, `--show-full-cmd`   : Display full command and cpu stats of running process
* `-p`, `--show-pid`   : Display PID of the process
* `-F`, `--show-fan`   : Display GPU fan speed
* `-e`, `--show-codec` : Display encoder and/or decoder utilization
* `-P`, `--show-power` : Display GPU power usage and/or limit (`draw` or `draw,limit`)
* `-a`, `--show-all`   : Display all gpu properties above
* `--no-processes`    : Do not display process information (user, memory)
* `--watch`, `-i`, `--interval`   : Run in watch mode (equivalent to `watch gpustat`) if given. Denotes interval between updates. ([#41][gh-issue-41])
* `--json`             : JSON Output (Experimental, [#10][gh-issue-10])
* `--print-completion (bash|zsh|tcsh)` : Print a shell completion script. See [#131][gh-issue-131] for usage.


### Tips

- Try `gpustat --debug` if something goes wrong.
- To periodically watch, try `gpustat --watch` or `gpustat -i` ([#41][gh-issue-41]).
    - For older versions, one may use `watch --color -n1.0 gpustat --color`.
- Running `nvidia-smi daemon` (root privilege required) will make querying GPUs much **faster** and use less CPU ([#54][gh-issue-54]).
- The GPU ID (index) shown by `gpustat` (and `nvidia-smi`) is PCI BUS ID,
  while CUDA uses a different ordering (assigns the fastest GPU with the lowest ID) by default.
  Therefore, in order to ensure CUDA and `gpustat` use **same GPU index**,
  configure the `CUDA_DEVICE_ORDER` environment variable to `PCI_BUS_ID`
  (before setting `CUDA_VISIBLE_DEVICES` for your CUDA program):
  `export CUDA_DEVICE_ORDER=PCI_BUS_ID`.


[pypi_gpustat]: https://pypi.org/project/gpustat/
[pypi_pynvml]: https://pypi.org/project/nvidia-ml-py/#history
[gh-issue-10]: https://github.com/wookayin/gpustat/issues/10
[gh-issue-41]: https://github.com/wookayin/gpustat/issues/41
[gh-issue-54]: https://github.com/wookayin/gpustat/issues/54
[gh-issue-66]: https://github.com/wookayin/gpustat/issues/66
[gh-issue-107]: https://github.com/wookayin/gpustat/issues/107
[gh-issue-131]: https://github.com/wookayin/gpustat/issues/131

Default display
---------------

```
[0] GeForce GTX Titan X | 77°C,  96 % | 11848 / 12287 MB | python/52046(11821M)
```

- `[0]`: GPU index (starts from 0) as PCI_BUS_ID
- `GeForce GTX Titan X`: GPU name
- `77°C`: GPU Temperature (in Celsius)
- `96 %`: GPU Utilization
- `11848 / 12287 MB`: GPU Memory Usage (Used / Total)
- `python/...`: Running processes on GPU, owner/cmdline/PID (and their GPU memory usage)

Changelog
---------

See [CHANGELOG.md](CHANGELOG.md)


License
-------

[MIT License](LICENSE)
