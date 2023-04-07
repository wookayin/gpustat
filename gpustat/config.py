"""
Defines a dataclass `PrintConfig` to handle options for printing gpustat to the
terminal, such as colors, other modifiers and overall arangement
"""

import re
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Dict, Literal, Optional, TypeVar, Union

from blessed import Terminal
# from omegaconf import OmegaConf


@dataclass
class ConditionalFormat:
    # TODO: Allow refernce to something else?
    mode: Literal["Larger", "Smaller", "Equal"]
    val: Union[int, float, str]
    eval_true: str
    eval_false: str
    eval_error: str = "normal"

StrPlus = TypeVar("StrPlus", str, ConditionalFormat)

def str_to_term(s: StrPlus, terminal: Terminal) -> StrPlus:
    """Converts a color entry below from human readable form to a ANSI-escape
    code of the specific terminal"""
    if isinstance(s, ConditionalFormat):
        s.eval_true = str_to_term(s.eval_true, terminal)
        s.eval_false = str_to_term(s.eval_false, terminal)
        s.eval_error = str_to_term(s.eval_error, terminal)
        return s

    code = terminal.normal # reset previous color/style
    for part in s.split(";"):
        if hasattr(terminal, part):
            code += getattr(terminal, part)
        else:
            try:
                # TODO: background modifier ?
                r, g, b = map(int, part.split(","))
                code += terminal.color_rgb(r, g, b)
            except:
                raise ValueError(f"Unknown color/style modifier: {part}")
    return code

@dataclass
class ProcecessFontModifiers:
    """Set colors for each kind of metric of a process running on a gpu"""
    username: str = "bold;black" # change here for readable dark terminals
    command: str = "blue"
    # TODO: full_command colors command uses two colors
    full_command: str = "cyan"
    gpu_memory_usage: str = "yellow"
    cpu_percent: str = "green"
    cpu_memory_usage: str = "yellow"
    pid: str = "normal"

    def to_term(self, terminal: Terminal):
        for var in fields(self):
            value = getattr(self, var.name)
            if not isinstance(value, (str, ConditionalFormat)):
                continue
            modifier = str_to_term(value, terminal)
            setattr(self, var.name, modifier)


@dataclass
class GPUFontModifiers:
    """Set colors for each kind of metric of a gpu"""
    # NOTE: dots in json are replaced with underscores here
    index: str = "cyan"
    uuid: str = "normal" # Not a default option
    name: str = "blue"
    temperature_gpu: str = "red" # TODO: is Conditional
    fan_speed: str = "cyan" # TODO: is Conditional
    utilization_gpu: str = "green" # TODO: is Conditional
    utilization_enc: str = "green" # TODO: is Conditional
    utilization_dec: str = "green" # TODO: is Conditional
    power_draw: str = "magenta" # TODO: is Conditional
    enforced_power_limit: str = "magenta"
    memory_used: str = "yellow"
    memory_total: str = "yellow"
    processes_font_modifiers: ProcecessFontModifiers = ProcecessFontModifiers()

    def to_term(self, terminal: Terminal):
        self.processes_font_modifiers.to_term(terminal)
        for var in fields(self):
            value = getattr(self, var.name)
            if not isinstance(value, (str, ConditionalFormat)):
                continue
            modifier = str_to_term(value, terminal)
            setattr(self, var.name, modifier)

@dataclass
class FontModifiers:
    """Set colors for each kind of all metric indepentendly"""
    # NOTE: Should this be overwritable in the string? Seems useless tbh
    gpu_font_modifiers: GPUFontModifiers = GPUFontModifiers()
    hostname: str = "bold;gray"
    driver_version: str = "bold;black"
    query_time: str = "normal"

    def to_term(self, terminal: Terminal):
        self.gpu_font_modifiers.to_term(terminal)
        for var in fields(self):
            value = getattr(self, var.name)
            if not isinstance(value, (str, ConditionalFormat)):
                continue
            modifier = str_to_term(value, terminal)
            setattr(self, var.name, modifier)

@dataclass
class PrintConfig:
    font_modifiers: FontModifiers = FontModifiers()
    # NOTE: Introduce color reset shortcut?

    # # Config 1: Roughly default `gpustat`
    # header: Optional[str] = "$hostname:{width}$ $query_time$ $driver_version$\n{gpus}"
    # gpus: Optional[str] = "[$index$] $name:{width}$ | $temperature_gpu:2${t.red}°C{t.normal}, $utilization_gpu:3$ {t.green}%{t.normal} | $memory_used:5$ / $memory_total:5$ MB | {processes}"
    # process: Optional[str] = "$username$ ($gpu_memory_usage$ MB)"
    # gpu_sep: str = "\n"
    # processes_sep: str = ", "

    # Config 2: Boxed config
    header: Optional[str] = (
        "┌{empty:─>{width}}{empty:─>39}┐\n"
        "│ $hostname:{width}$ $query_time$ $driver_version:>11$ │\n"
        "{gpus}\n"
        "└{empty:─>{width}}{empty:─>39}┘"
    )
    gpus: Optional[str] = "│ [$index$] $name:{width}$ │ $temperature_gpu:2${t.red}°C{t.normal}, $utilization_gpu:3$ {t.green}%{t.normal} │ $memory_used:5$ / $memory_total:5$ MB │{processes}"
    process: Optional[str] = "\n│     $username:>{width}$ │ $gpu_memory_usage$ MB {empty: >22}│"
    gpu_sep: str = "\n"
    processes_sep: str = ""

    gpuname_width: Optional[int] = None
    use_color: Optional[bool] = None
    extra_colors: Dict[str, str] = field(default_factory=dict) # dict()

    def __post_init__(self):
        def inner_prepare(s):
            return re.sub(r"\$([^:$]*?)(:[^:$]*?)?\$", r"{mods.\1}{\1\2}{t.normal}", s)
        self.header = inner_prepare(self.header)
        self.gpus = inner_prepare(self.gpus)
        self.process = inner_prepare(self.process)

    def to_term(self, terminal: Terminal):
        cp = deepcopy(self)
        for key, color in cp.extra_colors.values():
            cp.extra_colors[key] = str_to_term(color, terminal)
        cp.font_modifiers.to_term(terminal)
        return cp

    # def to_yaml(self) -> str:
    #     return OmegaConf.to_yaml(self)
