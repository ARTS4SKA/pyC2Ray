# ===================================================================================================
# This module manages the initialization of the ASORA raytracing extension library. It ensures that
# GPU memory has been allocated when GPU-accelerated functions are called.
# ===================================================================================================

import numpy as np

from pyc2ray.load_extensions import libasora

__all__ = [
    "average_fraction_to_device",
    "average_fraction_to_host",
    "check_device_init",
    "density_to_device",
    "device_close",
    "device_init",
    "flat_contiguous",
    "is_device_init",
    "is_periodic_mode_active",
    "photo_table_to_device",
    "source_data_to_device",
    "timestep_data_to_device",
]

# This flag indicates whether GPU memory has been correctly allocated before calling any methods.
# NOTE: there is no check if the allocated memory has the correct mesh size when calling a function,
# so the user is responsible for that.


def check_libasora(func):
    def _run_func(*args, **kwargs):
        if libasora is None:
            raise RuntimeError("ASORA Library not loaded")
        return func(*args, **kwargs)

    return _run_func


def check_device_init(func):
    @check_libasora
    def _run_func(*args, **kwargs):
        if not libasora.is_device_init():
            raise RuntimeError(
                "GPU not initialized. Please initialize it by calling device_init"
            )
        return func(*args, **kwargs)

    return _run_func


@check_libasora
def is_device_init() -> bool:
    assert libasora is not None
    return libasora.is_device_init()


@check_libasora
def is_periodic_mode_active() -> bool:
    """Return whether libasora was compiled with periodic boundary conditions"""
    assert libasora is not None
    return libasora.is_periodic_mode_active()


@check_libasora
def device_init(rank: int) -> None:
    """Initialize GPU and allocate memory for grid data

    Parameters
    ----------
    rank : int
        MPI rank of this process
    """
    assert libasora is not None
    libasora.device_init(rank)


@check_libasora
def device_close() -> None:
    """Deallocate GPU memory"""
    assert libasora is not None
    if libasora.is_device_init():
        libasora.device_close()


def flat_contiguous(arr: np.ndarray, dtype: type | None = None) -> np.ndarray:
    """Return a flattened, contiguous version of the input array."""
    return np.ascontiguousarray(arr, dtype=dtype).ravel()


@check_device_init
def density_to_device(density: np.ndarray) -> None:
    """Copy the density field to the GPU"""
    assert libasora is not None
    libasora.density_to_device(flat_contiguous(density))


@check_device_init
def source_data_to_device(
    source_pos: np.ndarray,
    source_flux: np.ndarray,
) -> None:
    """Copy the source data to the GPU"""
    assert libasora is not None
    libasora.source_data_to_device(
        flat_contiguous(source_pos), flat_contiguous(source_flux)
    )


@check_device_init
def photo_table_to_device(thin_table: np.ndarray, thick_table: np.ndarray) -> None:
    """Copy radiation tables to GPU (optically thin & thick tables)"""
    assert libasora is not None
    libasora.photo_table_to_device(
        flat_contiguous(thin_table), flat_contiguous(thick_table)
    )


@check_device_init
def timestep_data_to_device(
    xh: np.ndarray, temp: np.ndarray, clump: np.ndarray
) -> None:
    """Copy the fields that stay constant over one timestep to the GPU.

    Call this once per timestep, before the raytracing/chemistry loop. The
    average ionized fraction is seeded on the device from ``xh``, since the two
    are equal at the start of a timestep, and is updated in place by every
    chemistry pass thereafter.

    Parameters
    ----------
    xh : 1D-array of dtype float
        Ionized fraction at the start of the timestep.
    temp : 1D-array of dtype float
        Gas temperature of each cell in K.
    clump : 1D-array of dtype float
        Clumping factor of each cell.
    """
    assert libasora is not None
    libasora.timestep_data_to_device(
        flat_contiguous(xh), flat_contiguous(temp), flat_contiguous(clump)
    )


@check_device_init
def average_fraction_to_device(xh_av: np.ndarray) -> None:
    """Copy the average ionized fraction to the GPU.

    Only needed on ranks that do not run the chemistry pass themselves and
    therefore receive the field by broadcast. Ranks that run chemistry already
    hold the current values on the device.
    """
    assert libasora is not None
    libasora.average_fraction_to_device(flat_contiguous(xh_av))


@check_device_init
def average_fraction_to_host(xh_av: np.ndarray) -> None:
    """Copy the average ionized fraction back from the GPU into ``xh_av``.

    Only needed on the rank that broadcasts the field to the others; otherwise
    it never has to leave the device.
    """
    assert libasora is not None
    libasora.average_fraction_to_host(flat_contiguous(xh_av))
