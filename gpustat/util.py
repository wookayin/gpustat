""" Miscellaneous Utilities. """

import os.path


def bytes2human(in_bytes):
    '''

    '''
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
    Prettify and colorlize a full command-line (given as list of strings),
    where command (basename) is highlighted in a different color.
    '''
    # cmdline: Iterable[str]
    if isinstance(cmdline, str):
        return cmdline
    assert cmdline

    command_p, command_b = os.path.split(cmdline[0])
    s = color_text + os.path.join(command_p, color_command + command_b + color_text)

    if len(cmdline) > 1:
        s += ' '
        s += ' '.join(cmdline[1:])
    return s
