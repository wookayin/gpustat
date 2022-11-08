---
name: Bug report
about: Create a bug report for gpustat
title: ''
labels: bug
assignees: ''

---

<!-- Feel free to change the order of sections if appropriate. -->

**Describe the bug**

A clear and concise description of what the bug is.


**Screenshots or Program Output**

Please provide the output of `gpustat --debug` and `nvidia-smi`. Or attach screenshots if applicable.

<!--
```
$ gpustat --debug
```

```
$ nvidia-smi
```
-->


**Environment information:**

 - OS: [e.g. Ubuntu 18.04 LTS]
 - NVIDIA Driver version: [Try `nvidia-smi` or `gpustat`]
 - The name(s) of GPU card: [can be omitted if screenshot attached]
 - gpustat version: `gpustat --version`
 - pynvml version: Please provide the output of `pip list | grep nvidia-ml` or `sha1sum $(python -c 'import pynvml; print(pynvml.__file__)')`


**Additional context**

Add any other context about the problem here.
