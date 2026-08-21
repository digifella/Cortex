"""Windows long-path handling.

Paths over 260 characters need the \\?\ prefix even when LongPathsEnabled=1.
A PowerShell pass without it undercounted one P: folder by 47 GB, so every
filesystem call in this package goes through win_long().

On non-Windows these are identity functions, which is what lets the test
suite run under WSL.
"""
import os

LONG_PREFIX = "\\\\?\\"


def win_long(path: str) -> str:
    if os.name != "nt" or path.startswith(LONG_PREFIX):
        return path
    return LONG_PREFIX + os.path.abspath(path)


def strip_long(path: str) -> str:
    return path[len(LONG_PREFIX):] if path.startswith(LONG_PREFIX) else path
