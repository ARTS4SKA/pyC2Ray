"""Centralized place to load Fortran and C++/CUDA extensions for pyC2Ray"""

import warnings
from types import ModuleType

import pyc2ray.lib.libc2ray as libc2ray

try:
    libasora: ModuleType | None
    import pyc2ray.lib.libasora as libasora
except ImportError as e:
    warnings.warn(f"{e!s}. ASORA Library functionalities are disabled.")
    libasora = None

__all__ = ["libasora", "libc2ray"]
