from collections.abc import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyc2ray.asora_core import (
    average_fraction_to_device,
    average_fraction_to_host,
    check_libasora,
    density_to_device,
    device_close,
    device_init,
    flat_contiguous,
    is_device_init,
    is_periodic_mode_active,
    photo_table_to_device,
    source_data_to_device,
    timestep_data_to_device,
)
from pyc2ray.load_extensions import libasora

if libasora is None:
    pytest.skip("libasora.so missing, skipping tests", allow_module_level=True)


@check_libasora
def voidfunc() -> None: ...


@patch("pyc2ray.asora_core.libasora", new=None)
def test_check_libasora_missing() -> None:
    with pytest.raises(RuntimeError):
        voidfunc()


def test_check_libasora(init_device) -> None:
    voidfunc()


def test_device() -> None:
    assert not is_device_init()
    device_init(0)
    assert is_device_init()
    device_close()
    assert not is_device_init()


def test_is_periodic_mode_active() -> None:
    assert isinstance(is_periodic_mode_active(), bool)


def test_flat_contiguous_aliases_c_contiguous_input() -> None:
    """A flat, C-contiguous, float64 array is passed through as a view.

    Arrays the GPU writes into rely on this: the values must land in the caller's
    array, not in a temporary.
    """
    array = np.zeros(8, dtype=np.float64)
    flat_contiguous(array)[:] = 99.0
    assert np.all(array == 99.0)


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.zeros((2, 2, 2), dtype=np.float64, order="F"), id="fortran"),
        pytest.param(np.zeros(8, dtype=np.float64)[::2], id="strided"),
        pytest.param(np.zeros(8, dtype=np.float32), id="wrong-dtype"),
    ],
)
def test_flat_contiguous_copies_everything_else(array: np.ndarray) -> None:
    """Anything else is copied, so writes through the result are lost.

    This is correct for arrays the GPU only reads, and the reason an array the GPU
    writes into must already be flat, C-contiguous and float64 before it is passed
    down: see average_fraction_to_host.
    """
    flat_contiguous(array, dtype=np.float64)[:] = 99.0
    assert np.all(array == 0.0)


@pytest.fixture
def mock_libasora():
    """Replace the extension module with a mock that reports a live device.

    The wrappers look ``libasora`` up in the asora_core globals on every call, so
    replacing it there is what disarms the guards; patching the decorators would be
    too late, since they are applied at import time.
    """
    mock = Mock()
    mock.is_device_init.return_value = True
    with patch("pyc2ray.asora_core.libasora", mock):
        yield mock


def test_is_device_init_forwards_result(mock_libasora) -> None:
    mock_libasora.is_device_init.return_value = False
    assert is_device_init() is False
    mock_libasora.is_device_init.return_value = True
    assert is_device_init() is True


def test_device_init_forwards_rank(mock_libasora) -> None:
    device_init(3)
    mock_libasora.device_init.assert_called_once_with(3)


def test_device_close_when_initialized(mock_libasora) -> None:
    device_close()
    mock_libasora.device_close.assert_called_once_with()


def test_device_close_when_not_initialized(mock_libasora) -> None:
    """device_close is a no-op on an uninitialized device rather than an error.

    Callers use it as a teardown, so it has to be safe to run unconditionally.
    """
    mock_libasora.is_device_init.return_value = False
    device_close()
    mock_libasora.device_close.assert_not_called()


# Every wrapper that pushes or pulls arrays, with its entry point and arity.
TRANSFERS = [
    pytest.param(density_to_device, "density_to_device", 1, id="density"),
    pytest.param(source_data_to_device, "source_data_to_device", 2, id="source_data"),
    pytest.param(photo_table_to_device, "photo_table_to_device", 2, id="photo_table"),
    pytest.param(
        timestep_data_to_device, "timestep_data_to_device", 3, id="timestep_data"
    ),
    pytest.param(
        average_fraction_to_device, "average_fraction_to_device", 1, id="average_to_dev"
    ),
    pytest.param(
        average_fraction_to_host, "average_fraction_to_host", 1, id="average_to_host"
    ),
]


def make_args(count: int) -> list[np.ndarray]:
    """Return `count` distinguishable non-contiguous arrays, one per parameter."""
    return [
        np.full((2, 2, 2), value, dtype=np.float64, order="F")
        for value in range(1, count + 1)
    ]


@pytest.mark.parametrize(("wrapper", "entry_point", "count"), TRANSFERS)
def test_transfer_flattens_arguments(
    mock_libasora, wrapper: Callable, entry_point: str, count: int
) -> None:
    arrays = make_args(count)
    wrapper(*arrays)

    call = getattr(mock_libasora, entry_point).call_args
    assert len(call.args) == count
    for passed, original in zip(call.args, arrays, strict=True):
        assert passed.ndim == 1
        assert passed.flags.c_contiguous
        assert np.array_equal(passed, original.ravel())


@pytest.mark.parametrize(("wrapper", "entry_point", "count"), TRANSFERS)
def test_transfer_requires_device(
    mock_libasora, wrapper: Callable, entry_point: str, count: int
) -> None:
    mock_libasora.is_device_init.return_value = False

    with pytest.raises(RuntimeError, match="GPU not initialized"):
        wrapper(*make_args(count))

    getattr(mock_libasora, entry_point).assert_not_called()


@pytest.mark.parametrize(("wrapper", "entry_point", "count"), TRANSFERS)
def test_transfer_requires_library(
    wrapper: Callable, entry_point: str, count: int
) -> None:
    with (
        patch("pyc2ray.asora_core.libasora", new=None),
        pytest.raises(RuntimeError, match="not loaded"),
    ):
        wrapper(*make_args(count))
