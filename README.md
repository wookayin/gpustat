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

Just download [the script][script_gitio] into somewhere in `PATH`, e.g. `~/.local/bin/`:

```
sudo wget https://git.io/gpustat -O /usr/local/bin/gpustat && sudo chmod +x /usr/local/bin/gpustat
```

[script_gitio]: https://git.io/gpustat


Status and TODOs
----------------

This script is currently in infancy, so please be patient or give me a PR :-)

* [x] the list of Nvidia GPUs, along with temperature/utilization/memory statistics
* [x] the list of running processes: PIDs, memory usage, and process name/owner
* [x] basic ANSI color support
* [x] more configurable options
* [ ] sensible installation guide
