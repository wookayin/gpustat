""" Miscellaneous Utilities. """

import collections
import os.path
import sys
import traceback
from typing import Callable, Tuple, Type, TypeVar, Union

T = TypeVar('T')


def bytes2human(in_bytes):
    '''Convert bytes (int) to a human-readable string.'''
    suffixes = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
    suffix = 0
    result = int(in_bytes)
    while result > 9999 and suffix < len(suffixes):
        result = result >> 10
        suffix += 1

    if suffix >= len(suffixes):
        suffix -= 1
    return "%d%s" % (result, suffixes[suffix])


def prettify_commandline(cmdline, color_command='', color_text=''):
    '''
    Prettify and colorize a full command-line (given as list of strings),
    where command (basename) is highlighted in a different color.
    '''
    # cmdline: Iterable[str]
    if isinstance(cmdline, str):
        return cmdline
    assert cmdline

    command_p, command_b = os.path.split(cmdline[0])
    s = color_text + os.path.join(command_p,
                                  color_command + command_b + color_text)

    if len(cmdline) > 1:
        s += ' '
        s += ' '.join(cmdline[1:])
    return s


def shorten_left(text, width, placeholder="…"):
    # text: str
    if width is None:
        return text
    if text is None or len(text) <= width:
        return text
    if width < 0:
        raise ValueError("width must be non-negative.")
    if width == 0:
        return ""

    if width == len(placeholder):
        return placeholder
    elif width - len(placeholder) < 0:
        return placeholder[:width]
        # raise ValueError("width is smaller than the length of placeholder.")
    return placeholder + text[-(width - len(placeholder)):]


def safecall(fn: Callable[[], T],
             *,
             exc_types: Union[Type, Tuple[Type, ...]] = Exception,
             error_value: T) -> T:
    """A protected call that suppress certain types of exceptions."""
    try:
        return fn()
    except exc_types:  # pylint: disable=broad-except
        return error_value


class DebugHelper:

    def __init__(self):
        self._reports = []

    def add_exception(self, column, e=None):
        msg = "> An error while retrieving `{column}`: {e}".format(
            column=column, e=str(e))
        self._reports.append((msg, e))

    def _write(self, msg):
        sys.stderr.write(msg)
        sys.stderr.write('\n')

    def report_summary(self, concise=True):
        _seen_messages = collections.defaultdict(int)
        for msg, e in self._reports:
            if msg not in _seen_messages or not concise:
                self._write(msg)
                self._write(''.join(
                    traceback.format_exception(None, e, e.__traceback__)))
            _seen_messages[msg] += 1

        if concise:
            for msg, value in _seen_messages.items():
                self._write("{msg} -> Total {value} occurrences.".format(
                    msg=msg, value=value))
            self._write('')
