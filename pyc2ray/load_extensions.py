"""Centralized place to load Fortran and C++/CUDA extensions for pyC2Ray"""

import warnings
from types import ModuleType

import pyc2ray.lib.libc2ray as libc2ray

libasora: ModuleType | None
libasora_He: ModuleType | None

try:
    import pyc2ray.lib.libasora as libasora  # type: ignore
    import pyc2ray.lib.libasora_He as libasora_He  # type: ignore
except ImportError as e:
    print("Import error!")
    warnings.warn(f"{e!s}. ASORA Library functionalities are disabled.")
    libasora = None
    libasora_He = None

__all__ = ["libasora", "libasora_He", "libc2ray"]
