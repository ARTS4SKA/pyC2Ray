"""This file contains the main time-evolution subroutine, which updates
the ionization state of the whole grid over one timestep, using the
C2Ray method.

The raytracing step can use either the sequential (subbox, cubic)
technique which runs in Fortran on the CPU or the accelerated technique,
which runs using the ASORA library on the GPU.

When using the latter, some notes apply:
For performance reasons, the program minimizes the frequency at which
data is moved between the CPU and the GPU (this is a big bottleneck).
In particular, the radiation tables, which in principle shouldn't change
over the run of a simulation, need to be copied separately to the GPU
using the photo_table_to_device() method of the module. This is done
automatically when using the C2Ray subclasses but must be done manually
if for some reason you are calling the evolve3D routine directly without
using the C2Ray subclasses.

This file defines two variants of evolve3D: The reference, single-gpu
version, and a MPI version which enables usage on multiple GPU nodes.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from pyc2ray.asora_core import is_device_init
from pyc2ray.load_extensions import libasora_He as libasora
from pyc2ray.load_extensions import libc2ray
from pyc2ray.utils.logutils import allow_rank_logging
from pyc2ray.utils.other_utils import display_seconds, distribute_jobs
from pyc2ray.utils.sourceutils import FloatArray, IntArray

__all__ = ["evolve3D"]

logger = logging.getLogger(__name__)


@dataclass
class ChemistryParams:
    """Physics constants/parameters used by the chemistry solver.

    Parameters
    ----------
    bh00
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation.
    albpow
        Power-law index for the H recombination parameter.
    colh0
        Hydrogen collisional ionization parameter.
    temph0
        Hydrogen ionization energy expressed in K.
    abu_c
        Carbon abundance.
    """

    bh00: float
    albpow: float
    colh0: float
    temph0: float
    abu_c: float


def relative_change(old: float, new: float) -> float:
    """Compute the relative change between an old and new value, with safe handling of zero values."""
    return abs((new - old) / new) if new > 0.0 else 1.0


def _evolve3D_asora(
    dt: float,
    dr: float,
    src_flux: FloatArray,
    src_pos: IntArray,
    src_batch_size: int,
    max_subbox: int,
    subboxsize: int,
    loss_fraction: float,
    use_mpi: bool,
    rank: int,
    nprocs: int,
    temp: FloatArray,
    ndens: FloatArray,
    xh: FloatArray,
    clump: FloatArray,
    photo_thin_table: FloatArray,
    photo_thick_table: FloatArray,
    minlogtau: float,
    dlogtau: float,
    R_max_LLS: float,
    convergence_fraction: float,
    sigma: float,
    chems: ChemistryParams,
) -> tuple[FloatArray, FloatArray]:
    """Evolves the ionization fraction over one timestep for the whole grid

    Warning: Calling this function assumes that the radiation tables have previously been
    copied to the GPU using photo_table_to_device()

    Parameters
    ----------
    dt
        Timestep in seconds
    dr
        Cell dimension in each direction in cm.
    src_flux
        Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default).
    src_pos
        Array containing the 3D grid position of each source, in Fortran indexing (from 1).
    max_subbox
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true.
    subboxsize
        ...
    loss_fraction
        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true.
    temp
        The initial temperature of each cell in K.
    ndens
        The hydrogen number density of each cell in cm^-3.
    xh
        The initial ionized fraction of each cell.
    photo_thin_table
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
        in a separate (previous) step, using photo_table_to_device().
    minlogtau
        Base 10 log of the minimum value of the table in τ (excluding τ = 0).
    dlogtau
        Step size of the logτ-table.
    R_max_LLS
        Value of maximum comoving distance for photons from source (type 3 LLS in original C2Ray). This value is
        given in cell units, but doesn't need to be an integer.
    convergence_fraction
        Which fraction of the cells can be left unconverged to improve performance (usually ~ 1e-4).
    sigma
        Constant photoionization cross-section of hydrogen in cm^2.
    chems
        Parameters used by the chemistry solver.

    Returns
    -------
    xh_int : 3D-array of dtype float
        The updated ionization fraction of each cell at the end of the timestep.
    phi_ion : 3D-array of dtype float
        Photoionization rate of each cell due to all sources.
    """
    if not is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )

    rank_prefix = f"[Rank={rank}] " if use_mpi else ""

    # Problem dimensions.
    N, _, _ = mesh_shape = ndens.shape
    num_cells = np.prod(mesh_shape)
    num_src, *_ = src_flux.shape
    num_tau, *_ = photo_thin_table.shape

    # Convergence Criteria
    conv_criterion = min(int(convergence_fraction * num_cells), (num_src - 1) / 3)
    prev_sum_xh1 = float(2 * num_cells)
    prev_sum_xh0 = float(2 * num_cells)
    converged = False

    logger.info(f"""Calling evolve3D...
dr [Mpc]: {dr / 3.086e24:.3e}
dt [years]: {dt / 3.15576e07:.3e}
Running on {num_src:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}
Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: {xh.mean():.3e}
Convergence Criterion (Number of points): {conv_criterion: n}
""")

    # Prepare source data for GPU. If using MPI, use a subset of sources for each rank.
    src_pos = src_pos.astype(np.int32)
    src_flux = src_flux.astype(np.float64)
    if use_mpi:
        chunk = distribute_jobs(num_src, nprocs, rank)

        # Overwrite number of sources
        num_src = chunk.stop - chunk.start
        src_pos = src_pos[:, chunk]
        src_flux = src_flux[chunk]

        with allow_rank_logging(rank):
            logger.info(f"{rank_prefix}{num_src} sources.")

    # Copy positions & fluxes of sources to the GPU
    assert libasora is not None
    libasora.source_data_to_device(src_pos.ravel(), src_flux.ravel())

    # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
    assert libasora is not None
    ndens_flat = np.ravel(ndens).astype(np.float64)
    libasora.density_to_device(ndens_flat)

    # Initialize average and intermediate results
    xh_flat = np.ravel(xh).astype(np.float64)
    xh_av = xh_flat.copy()
    xh_int = xh_flat.copy()

    # Initialize ionization rate array.
    phi_ion = np.zeros_like(ndens_flat)

    # Prepare other inputs
    temp_flat = np.ravel(temp).astype(np.float64)
    clump_flat = np.ravel(clump).astype(np.float64)

    with allow_rank_logging(rank):
        logger.info(f"{rank_prefix}Copied source data to device.")

    if rank == 0:
        # Iteration counter
        n_count = 0

    while not converged:
        # --------------------
        # (1): Raytracing Step
        # --------------------
        with allow_rank_logging(rank):
            logger.info(f"{rank_prefix}Doing Raytracing...")

        time_start = time.perf_counter()

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        # This function updates phi_ion.
        assert libasora is not None
        libasora.do_all_sources(
            R_max_LLS,
            sigma,
            dr,
            xh_av,
            phi_ion,
            num_src,
            N,
            minlogtau,
            dlogtau,
            num_tau,
            src_batch_size,
        )

        time_end = time.perf_counter()
        with allow_rank_logging(rank):
            logger.info(
                f"{rank_prefix}...took {display_seconds(time_end - time_start)}"
            )

        if use_mpi:
            # Collect results from the different MPI processors
            if rank == 0:
                MPI.COMM_WORLD.Reduce(
                    MPI.IN_PLACE, [phi_ion, MPI.DOUBLE], op=MPI.SUM, root=0
                )
            else:
                MPI.COMM_WORLD.Reduce([phi_ion, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            MPI.COMM_WORLD.Bcast([phi_ion, MPI.DOUBLE], root=0)

        if rank == 0:
            # ---------------------
            # (2): ODE Solving Step  TODO: Split this to MPI ranks
            # ---------------------
            logger.info("Doing Chemistry...")

            time_start = time.perf_counter()

            # Apply the global rates to compute the updated ionization fraction
            # This function updates xh_av and xh_int.
            conv_flag = libasora.chemistry_global_pass(
                dt,
                temp_flat,
                xh_flat,
                xh_av,
                xh_int,
                phi_ion,
                clump_flat,
                chems.bh00,
                chems.albpow,
                chems.colh0,
                chems.temph0,
                chems.abu_c,
            )

            time_end = time.perf_counter()
            logger.info(f"  took {display_seconds(time_end - time_start)}")

            # ----------------------------
            # (3): Test Global Convergence
            # ----------------------------
            sum_xh1 = xh_int.sum()
            sum_xh0 = xh_int.size - sum_xh1  # = np.sum(1 - xh_int)

            rel_change_xh1 = relative_change(prev_sum_xh1, sum_xh1)
            rel_change_xh0 = relative_change(prev_sum_xh0, sum_xh0)

            # Display convergence
            logger.info(
                f"Number of non-converged points: {conv_flag} of {num_cells} ({conv_flag / num_cells * 100: .3f} % ), "
                f"Relative change in ionfrac: {rel_change_xh1: .2e}",
            )

            converged = (conv_flag < conv_criterion) or (
                (rel_change_xh1 < convergence_fraction)
                and (rel_change_xh0 < convergence_fraction)
            )
            n_count += 1

            # Set previous metrics to current ones and repeat if not converged
            prev_sum_xh1 = sum_xh1
            prev_sum_xh0 = sum_xh0

        if use_mpi:
            # broadcast ionised fraction field
            MPI.COMM_WORLD.Bcast([xh_av, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xh_int, MPI.DOUBLE], root=0)

            # broadcast convergence
            MPI.COMM_WORLD.bcast(converged, root=0)

    if rank == 0:
        logger.info(
            f"Multiple source convergence reached after {n_count} ray-tracing iterations."
        )

    return xh_int.reshape(mesh_shape), phi_ion.reshape(mesh_shape)


def _evolve3D_c2ray(
    dt: float,
    dr: float,
    src_flux: FloatArray,
    src_pos: IntArray,
    src_batch_size: int,
    max_subbox: int,
    subboxsize: int,
    loss_fraction: float,
    use_mpi: bool,
    rank: int,
    nprocs: int,
    temp: FloatArray,
    ndens: FloatArray,
    xh: FloatArray,
    clump: FloatArray,
    photo_thin_table: FloatArray,
    photo_thick_table: FloatArray,
    minlogtau: float,
    dlogtau: float,
    R_max_LLS: float,
    convergence_fraction: float,
    sigma: float,
    chems: ChemistryParams,
) -> tuple[FloatArray, FloatArray]:
    """Evolves the ionization fraction over one timestep for the whole grid

    Parameters
    ----------
    dt
        Timestep in seconds
    dr
        Cell dimension in each direction in cm.
    src_flux
        Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default).
    src_pos
        Array containing the 3D grid position of each source, in Fortran indexing (from 1).
    max_subbox
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true.
    subboxsize
        ...
    loss_fraction
        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true.
    temp
        The initial temperature of each cell in K.
    ndens
        The hydrogen number density of each cell in cm^-3.
    xh
        The initial ionized fraction of each cell.
    photo_thin_table
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
        in a separate (previous) step, using photo_table_to_device().
    minlogtau
        Base 10 log of the minimum value of the table in τ (excluding τ = 0).
    dlogtau
        Step size of the logτ-table.
    R_max_LLS
        Value of maximum comoving distance for photons from source (type 3 LLS in original C2Ray). This value is
        given in cell units, but doesn't need to be an integer.
    convergence_fraction
        Which fraction of the cells can be left unconverged to improve performance (usually ~ 1e-4).
    sigma
        Constant photoionization cross-section of hydrogen in cm^2.
    chems
        Parameters used by the chemistry solver.

    Returns
    -------
    xh_int : 3D-array of dtype float
        The updated ionization fraction of each cell at the end of the timestep.
    phi_ion : 3D-array of dtype float
        Photoionization rate of each cell due to all sources.
    """
    rank_prefix = f"Rank={rank}: " if use_mpi else ""

    # Problem dimensions.
    N, _, _ = mesh_shape = ndens.shape
    num_cells = np.prod(mesh_shape)
    num_src, *_ = src_flux.shape
    num_tau, *_ = photo_thin_table.shape

    # Convergence criteria.
    conv_criterion = min(int(convergence_fraction * num_cells), (num_src - 1) / 3)
    prev_sum_xh1 = float(2 * num_cells)
    prev_sum_xh0 = float(2 * num_cells)
    converged = False

    logger.info(f"""Calling evolve3D...
dr [Mpc]: {dr / 3.086e24:.3e}
dt [years]: {dt / 3.15576e07:.3e}
Running on {num_src:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}
Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: {xh.mean():.3e}
Convergence Criterion (Number of points): {conv_criterion: n}
""")

    # Prepare source data. If using MPI, use a subset of sources for each rank.
    src_pos = src_pos.astype(np.int32, order="F")
    src_flux = src_flux.astype(np.float64, order="F")
    if use_mpi:
        chunk = distribute_jobs(num_src, nprocs, rank)

        # Overwrite number of sources
        num_src = chunk.stop - chunk.start
        src_pos = src_pos[:, chunk]
        src_flux = src_flux[chunk]

        with allow_rank_logging(rank):
            logger.info(f"{rank_prefix}{num_src} sources.")

    # Initialize average and intermediate results.
    xh_av = xh.copy(order="F")
    xh_int = xh.copy(order="F")

    # Placeholder, eventually we'll add heating tables here
    heat_thin_table = np.zeros_like(photo_thin_table)
    heat_thick_table = np.zeros_like(photo_thick_table)

    if rank == 0:
        # Iteration counter
        n_count = 0

    while not converged:
        # --------------------
        # (1): Raytracing Step
        # --------------------
        with allow_rank_logging(rank):
            logger.info(f"{rank_prefix}Doing Raytracing...")

        time_start = time.perf_counter()

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        phi_ion = np.zeros(mesh_shape, dtype=np.float64, order="F")
        phi_heat = np.zeros_like(phi_ion)
        coldensh_out = np.zeros_like(phi_ion)

        # Use CPU raytracing with subbox optimization
        nsubbox, photonloss = libc2ray.raytracing.do_all_sources(
            src_flux,
            src_pos,
            max_subbox,
            subboxsize,
            coldensh_out,
            sigma,
            dr,
            ndens,
            xh_av,
            phi_ion,
            phi_heat,
            loss_fraction,
            photo_thin_table,
            photo_thick_table,
            heat_thin_table,
            heat_thick_table,
            minlogtau,
            dlogtau,
            R_max_LLS,
        )

        time_end = time.perf_counter()
        with allow_rank_logging(rank):
            logger.info(f"  took {display_seconds(time_end - time_start)}")

        logger.info(
            f"Average number of subboxes: {nsubbox / num_src:n}, Total photon loss: {photonloss:.3e}"
        )

        if use_mpi:
            # Collect results from the different MPI processors
            if rank == 0:
                MPI.COMM_WORLD.Reduce(
                    MPI.IN_PLACE, [phi_ion, MPI.DOUBLE], op=MPI.SUM, root=0
                )
            else:
                MPI.COMM_WORLD.Reduce([phi_ion, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            MPI.COMM_WORLD.Bcast([phi_ion, MPI.DOUBLE], root=0)

        if rank == 0:
            # ---------------------
            # (2): ODE Solving Step
            # ---------------------
            logger.info("Doing Chemistry...")

            time_start = time.perf_counter()

            # Apply the global rates to compute the updated ionization fraction
            conv_flag = libc2ray.chemistry.global_pass(
                dt,
                ndens,
                temp,
                xh,
                xh_av,
                xh_int,
                phi_ion,
                clump,
                chems.bh00,
                chems.albpow,
                chems.colh0,
                chems.temph0,
                chems.abu_c,
            )

            time_end = time.perf_counter()
            logger.info(f"...took {display_seconds(time_end - time_start)}")

            # ----------------------------
            # (3): Test Global Convergence
            # ----------------------------
            sum_xh1 = xh_int.sum()
            sum_xh0 = xh_int.size - sum_xh1  # = np.sum(1 - xh_int)

            rel_change_xh1 = relative_change(prev_sum_xh1, sum_xh1)
            rel_change_xh0 = relative_change(prev_sum_xh0, sum_xh0)

            # Display convergence
            logger.info(
                f"Number of non-converged points: {conv_flag} of {num_cells} ({conv_flag / num_cells:.3%}), "
                f"Relative change in ionfrac: {rel_change_xh1:.2e}",
            )

            converged = (conv_flag < conv_criterion) or (
                (rel_change_xh1 < convergence_fraction)
                and (rel_change_xh0 < convergence_fraction)
            )
            n_count += 1

            # Set previous metrics to current ones and repeat if not converged
            prev_sum_xh1 = sum_xh1
            prev_sum_xh0 = sum_xh0

        if use_mpi:
            # broadcast ionised fraction field
            MPI.COMM_WORLD.Bcast([xh_av, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xh_int, MPI.DOUBLE], root=0)

            # broadcast convergence
            converged = MPI.COMM_WORLD.bcast(converged, root=0)

    if rank == 0:
        logger.info(
            f"Multiple source convergence reached after {n_count} ray-tracing iterations."
        )

    return xh_int, phi_ion


def evolve3D(**kwargs) -> tuple[FloatArray, FloatArray]:
    """Evolves the ionization fraction over one timestep for the whole grid
    Relays the call to either evolve3D_asora or evolve3D_c2ray depending on the value of use_gpu in the arguments.
    """
    # TODO: factorize common code between the two versions
    if kwargs.pop("use_gpu", False):
        if libasora is not None:
            return _evolve3D_asora(**kwargs)
        logger.warning(
            "Required GPU computation in evolve but ASORA library is not available. Falling back to CPU computation."
        )
    return _evolve3D_c2ray(**kwargs)
