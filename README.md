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

* `--no-color`        : Suppress color (by default, color is enabled)
* `-u`, `--show-user` : Display username of the process owner
* `-c`, `--show-cmd`  : Display the process name
* `-p`, `--show-pid`  : Display PID of the process
* `--json`            : JSON Output (Experimental, #10)

### Tips

- To periodically watch, try `watch --color -n1.0 gpustat` (built-in watch support will be added soon).
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
sudo pip install gpustat
```

To install the latest version (master branch) via pip:

```
pip install git+https://github.com/wookayin/gpustat.git@master
```

If you don't have root privilege, you can try `pip install --user` as well.
Please note that from v0.4, `gpustat.py` is no more a zero-dependency executable.

```
sudo wget https://git.io/gpustat.py -O /usr/local/bin/gpustat && sudo chmod +x /usr/local/bin/gpustat
```

[pypi_gpustat]: https://pypi.python.org/pypi/gpustat


License
-------

[MIT License](LICENSE)
