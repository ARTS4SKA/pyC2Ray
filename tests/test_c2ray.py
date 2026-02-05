from pyc2ray.load_extensions import libc2ray


def test_load_c2ray():
    assert libc2ray is not None
