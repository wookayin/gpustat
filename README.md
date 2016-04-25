`gpustat`
=========

Just *less* than nvidia-smi?

Usage
-----

`$ gpustat`

A possible output:

```
[0] GeForce GTX TITAN X | 71'C,  39 % | Mem: 11721 / 12287 MB | python(11696M)
[1] GeForce GTX TITAN X | 70'C,  84 % | Mem: 11833 / 12287 MB | matrixMul(112M) python(11696M)
[2] GeForce GTX TITAN X | 38'C,   0 % | Mem:    23 / 12287 MB |
```

Quick Installation
------------------

Just download [the script][script_gitio] into somewhere in `PATH`, e.g. `~/.local/bin/`:

```
sudo curl https://git.io/gpustat -O /usr/local/bin/gpustat
sudo chmod +x /usr/local/bin/gpustat
```

[script_gitio]: https://git.io/gpustat

Status and TODOs
----------------

This script is currently in infancy, so please be patient or give me a PR :-)

* [x] the list of Nvidia GPUs, along with temperature/utilization/memory statistics
* [x] the list of running processes: PIDs, memory usage, and process name/owner
* [ ] more configurable options
* [ ] sensible installation guide
