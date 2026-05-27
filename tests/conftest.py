from pathlib import Path

import pytest

from pyc2ray.load_extensions import libasora


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
    if libasora is not None:
        libasora.device_init()
    yield
    if libasora is not None:
        libasora.device_close()
