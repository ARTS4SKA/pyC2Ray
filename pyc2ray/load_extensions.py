"""Centralized place to load Fortran and C++/CUDA extensions for pyC2Ray"""

import warnings
from types import ModuleType

from pyc2ray.lib import libc2ray

try:
    libasora: ModuleType | None
    from pyc2ray.lib import libasora
except ImportError as e:
    warnings.warn(f"{e!s}. ASORA Library functionalities are disabled.")
    libasora = None

__all__ = ["libasora", "libc2ray"]
