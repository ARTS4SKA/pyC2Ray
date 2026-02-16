from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the test folder"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(test_dir: Path) -> Path:
    """Return the path to the data folder for tests"""
    return test_dir / "data"


@pytest.fixture
def init_device():
    try:
        from pyc2ray.lib import libasora
    except ImportError:
        return

    DEFAULT_GPU = 0
    libasora.device_init(DEFAULT_GPU)
    yield
    libasora.device_close(DEFAULT_GPU)
