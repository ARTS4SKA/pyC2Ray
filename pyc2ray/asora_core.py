# ===================================================================================================
# This module manages the initialization of the ASORA raytracing extension library. It ensures that
# GPU memory has been allocated when GPU-accelerated functions are called.
# ===================================================================================================

from .load_extensions import libasora

__all__ = ["is_device_init", "device_init", "device_close", "photo_table_to_device"]

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
def photo_table_to_device(thin_table, thick_table):
    """Copy radiation tables to GPU (optically thin & thick tables)"""
    assert libasora is not None
    if not libasora.is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init"
        )
    libasora.photo_table_to_device(thin_table, thick_table)
