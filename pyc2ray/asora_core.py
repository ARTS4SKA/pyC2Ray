# ===================================================================================================
# This module manages the initialization of the ASORA raytracing extension library. It ensures that
# GPU memory has been allocated when GPU-accelerated functions are called.
# ===================================================================================================

from .load_extensions import libasora

__all__ = [
    "is_device_init",
    "is_periodic_mode_active",
    "device_init",
    "device_close",
    "prepare_grid_buffers",
    "photo_table_to_device",
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


@check_libasora
def prepare_grid_buffers(
    grid_edge_length: int, force_matching_size: bool = False
) -> None:
    """Ensure mesh-dependent device buffers exist for an m1^3 grid

    Parameters
    ----------
    grid_edge_length : int
        Grid edge length; buffers are sized for grid_edge_length**3 cells
    force_matching_size : bool
        If True, reallocate buffers when their current size does not match
    """
    assert libasora is not None
    if not libasora.is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init"
        )
    libasora.prepare_grid_buffers(grid_edge_length, force_matching_size)


@check_libasora
def photo_table_to_device(thin_table, thick_table):
    """Copy radiation tables to GPU (optically thin & thick tables)"""
    assert libasora is not None
    if not libasora.is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init"
        )
    libasora.photo_table_to_device(thin_table, thick_table)
