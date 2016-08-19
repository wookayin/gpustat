`gpustat`
=========

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

### Tips

- To periodically watch, try `watch --color -n1.0 gpustat` (built-in watch support will be added soon).
- Running `nvidia-smi daemon` (root privilege required) will make the query much faster.


Quick Installation
------------------

Install from PyPI:

```
sudo pip install gpustat
```

Alternatively, you can just download [gpustat.py][script_gitio] into somewhere in `PATH`, e.g. `~/.local/bin/`
(when you do not have root privilege, for example):

```
sudo wget https://git.io/gpustat.py -O /usr/local/bin/gpustat && sudo chmod +x /usr/local/bin/gpustat
```

[script_gitio]: https://git.io/gpustat.py
