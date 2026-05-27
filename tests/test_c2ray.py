from pyc2ray.load_extensions import libc2ray


def test_load_c2ray() -> None:
    assert libc2ray is not None
    assert hasattr(libc2ray, "chemistry_he")
    assert hasattr(libc2ray.chemistry_he, "thermal")
    assert hasattr(libc2ray.chemistry_he, "global_pass")
