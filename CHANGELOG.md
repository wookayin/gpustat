Changelog for `gpustat`
=======================

## [v1.0][milestone-1.0] (2022/9/4)

[milestone-1.0]: https://github.com/wookayin/gpustat/issues?q=milestone%3A1.0

### Breaking Changes

- Retire Python 2 (#66). Add CI tests for python 3.8 and higher.
- Use official nvidia python bindings (#107).
    - Due to API incompatibility issues, the nvidia driver version should be **R450** or higher
      in order for process information to be correctly displayed.
    - NOTE: `nvidia-ml-py<=11.495.46` is required (`nvidia-ml-py3` shall not be used).
- Use of '--gpuname-width' will truncate longer GPU names (#47).

### New Feature and Enhancements

- Add windows support again, by switching to `blessed` (#78, @skjerns)
- Add '--show-codec (-e)' option: display encoder/decoder utilization (#79, @ChaoticMind)
- Add full process information (-f) (#65, @bethune-bryant)
- Add '--show-all (-a)' flag (#64)
- '--debug' will show more detailed stacktrace/exception information
- Use unicode symbols (#58, @arinbjornk)
- Include nvidia driver version into JSON output (#10)

### Bug Fixes

- Fix color/highlight issues on power usage
- Make color/highlight work correctly when TERM is not set
- Do not list the same GPU process more than once (#84)
- Fix a bug where querying zombie process can throw errors (#95)
- Fix a bug where psutil may fail to get process info on Windows (#121, #123, @mattip)

### Etc.

- Internal improvements on code style and tests
- CI: Use Github Actions


## [v0.6.0][milestone-0.6] (2019/07/22)

[milestone-0.6]: https://github.com/wookayin/gpustat/issues?q=milestone%3A0.6

- [Feature] Add a flag for fan speed (`-F`, `--show-fan`) (#62, #63), contributed by @bethune-bryant
- [Enhancement] Align query datetime in the header with respect to `--gpuname-width` parameter.
- [Enhancement] Alias `gpustat --watch` to `-i`/`--interval` option.
- [Enhancement] Display NVIDIA driver version in the header (#53)
- [Bugfix] Minor fixes on debug mode
- [Etc] Travis: python 3.7


## [v0.5.0][milestone-0.5] (2018/09/09)

[milestone-0.5]: https://github.com/wookayin/gpustat/issues?q=milestone%3A0.5

- [Feature] Built-in watch mode (`gpustat -i`) (#7, #41).
   - Contributed by @drons and @Stonesjtu, Thanks!
- [Bug] Fix the problem extra character was showing (#32)
- [Bug] Fix a bug in json mode where process information is unavailable (#45)
- [Etc.] Refactoring of internal code structure: `gpustat` is now a package (#33)
- [Etc.] More unit tests and better use of code styles (flake8)



## v0.4.1

- Fix a bug that might happen when power_draw is not available (#16)


## v0.4.0

`gpustat` is no more a zero-dependency script and now depends on some packages. Please install using pip.

- Use `nvidia-ml-py` bindings and `psutil` to replace command-line call of `nvidia-smi` and `ps` (#20, Thanks to @Stonesjtu).
- A behavior on pipe is changed; it will not be in color by default, use `--color` explicitly. (e.g. `watch --color -n1.0 gpustat --color`)
- Fix a bug in handling stale-state or zombie process (#16)
- Include non-CUDA graphics applications in the process list (#18, Thanks to @kapsh)
- Support power usage (#13, #28, Thanks to @cjw85)
- Support `--debug` option


## v0.3.1

- Experimental JSON output feature (#10)
- Add some properties and dict-style access for `GPUStat` class
- Fix Python3 compatibility


## v0.2.0

- Add `--gpuname-width` option
- Display long usernames correctly
- Support older NVIDIA cards (#6)
