"""Centralized place to load Fortran and C++/CUDA extensions for pyC2Ray"""

import warnings

import pyc2ray.lib.libasora as libasora
import pyc2ray.lib.libc2ray as libc2ray

try:
    from .lib import libasora
except ImportError as e:
    warnings.warn(f"{e!s}. ASORA Library functionalities are disabled.")
    libasora = None

__all__ = ["libasora", "libc2ray"]
