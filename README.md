`gpustat`
=========

[![pypi](https://img.shields.io/pypi/v/gpustat.svg?maxAge=86400)][pypi_gpustat]
[![Build Status](https://travis-ci.org/wookayin/gpustat.svg?branch=master)](https://travis-ci.org/wookayin/gpustat)
[![license](https://img.shields.io/github/license/wookayin/gpustat.svg?maxAge=86400)](LICENSE)

Just *less* than nvidia-smi?

![Screenshot: gpustat -cp](screenshot.png)

Usage
-----

`$ gpustat`

Options:

* `--color`            : Force colored output (even when stdout is not a tty)
* `--no-color`         : Suppress colored output
* `-u`, `--show-user`  : Display username of the process owner
* `-c`, `--show-cmd`   : Display the process name
* `-p`, `--show-pid`   : Display PID of the process
* `-P`, `--show-power` : Display GPU power usage and/or limit (`draw` or `draw,limit`)
* `--json`             : JSON Output (Experimental, [#10][gh-issue-10])

### Tips

- To periodically watch, try `watch --color -n1.0 gpustat --color` (built-in watch support will be added soon).
- Running `nvidia-smi daemon` (root privilege required) will make the query much **faster**.
- The GPU ID (index) shown by `gpustat` (and `nvidia-smi`) is PCI BUS ID,
  while CUDA differently assigns the fastest GPU with the lowest ID by default.
  Therefore, in order to make CUDA and `gpustat` use **same GPU index**,
  configure the `CUDA_DEVICE_ORDER` environment variable to `PCI_BUS_ID`
  (before setting `CUDA_VISIBLE_DEVICES` for your CUDA program):
  `export CUDA_DEVICE_ORDER=PCI_BUS_ID`


Quick Installation
------------------

Install from [PyPI][pypi_gpustat]:

```
pip install gpustat
```

To install the latest version (master branch) via pip:

```
pip install git+https://github.com/wookayin/gpustat.git@master
```

If you don't have root privilege, please try to install on user namespace: `pip install --user`.
Note that from v0.4, `gpustat.py` is no more a zero-dependency executable.
However, in rare cases you'll need a single executable script (legacy), you can also try:

```
wget https://raw.githubusercontent.com/wookayin/gpustat/v0.3.2/gpustat.py -O ~/.local/bin/gpustat
chmod +x ~/.local/bin/gpustat    # Assuming ~/.local/bin is in your $PATH
```

[pypi_gpustat]: https://pypi.python.org/pypi/gpustat
[gh-issue-10]: https://github.com/wookayin/gpustat/issues/10


License
-------

[MIT License](LICENSE)
