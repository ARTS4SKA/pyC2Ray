from unittest.mock import patch

import pytest

from pyc2ray.asora_core import check_libasora, device_close, device_init, is_device_init
from pyc2ray.load_extensions import libasora

if libasora is None:
    pytest.skip("libasora.so missing, skipping tests", allow_module_level=True)


@check_libasora
def voidfunc():
    return None


@patch("pyc2ray.asora_core.libasora", new=None)
def test_check_libasora_missing():
    with pytest.raises(RuntimeError):
        voidfunc()


def test_check_libasora():
    voidfunc()


def test_device():
    assert not is_device_init()
    device_init(0)
    assert is_device_init()
    device_close()
    assert not is_device_init()
