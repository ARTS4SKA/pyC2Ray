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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
from mpi4py import MPI

from pyc2ray.asora_core import is_device_init, libasora
from pyc2ray.load_extensions import libc2ray
from pyc2ray.radiation.radiation_tables import RadiationTables
from pyc2ray.utils.logutils import allow_rank_logging
from pyc2ray.utils.other_utils import display_seconds, distribute_jobs
from pyc2ray.utils.sourceutils import FloatArray, IntArray

__all__ = ["evolve3D"]

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD


@dataclass
class ChemistryParams:
    """Physics constants/parameters used by the chemistry solver.

    Parameters
    ----------
    bh00 :
        Hydrogen recombination parameter at 10^4 K in the case B OTS
        approximation.
    albpow :
        Power-law index for the H recombination parameter.
    colh0 :
        Hydrogen collisional ionization parameter.
    temph0 :
        Hydrogen ionization energy expressed in K.
    abu_c
        Carbon abundance.
    """

    bh00: float
    albpow: float
    colh0: float
    temph0: float
    abu_c: float


class FractionStats:
    """Hold the total neutral and ionized fractions of the different species acress the grid."""

    def __init__(self, data: Sequence[float]) -> None:
        """Initialize the total fraction stats with dummy values."""
        if len(data) != 6:
            raise ValueError(
                f"Expected 6 values for the total fraction stats, but got {len(data)}"
            )
        self.data = cast(tuple[float, float, float, float, float, float], tuple(data))

    @property
    def HII(self) -> tuple[float, float]:
        return self.data[0], self.data[1]

    @property
    def HeII(self) -> tuple[float, float]:
        return self.data[2], self.data[3]

    @property
    def HeIII(self) -> tuple[float, float]:
        return self.data[4], self.data[5]

    @classmethod
    def from_xh(self, *xh: FloatArray) -> FractionStats:
        """Compute the total ionized and neutral fraction of each species across the
        grid from the ionization fraction fields."""
        if len(xh) != 3:
            raise ValueError(
                f"Expected 3 ionization fraction fields for HII, HeII and HeIII, but got {len(xh)}"
            )
        tot_fracs: list[float] = []
        for frac in xh:
            tot = frac.sum()
            tot_fracs.append(frac.size - tot)
            tot_fracs.append(tot)
        return FractionStats(tot_fracs)

    def relative_change(self, xh_new: FractionStats) -> FractionStats:
        """Compute the relative change between an old and new value"""

        def _rel_change(old: float, new: float) -> float:
            return abs((new - old) / new) if new > 0.0 else 1.0

        return FractionStats(
            tuple(_rel_change(old, new) for old, new in zip(self.data, xh_new.data))
        )


def _evolve3D_asora(
    Hz: float,
    dt: float,
    dr: float,
    R_max: float,
    src_flux: FloatArray,
    src_pos: IntArray,
    src_batch_size: int,
    temp: FloatArray,
    ndens: FloatArray,
    clump: FloatArray,
    xh: tuple[FloatArray, FloatArray, FloatArray],
    chems: ChemistryParams,
    logtau: tuple[float, float, int],
    logtemp: tuple[float, float, int],
    convergence_fraction: float,
    use_mpi: bool,
    rank: int,
    nprocs: int,
    **kwargs,
) -> tuple[
    tuple[FloatArray, FloatArray, FloatArray],  # xfrac
    tuple[FloatArray, FloatArray, FloatArray],  # ion rate
    FloatArray,  # heat rate
    FloatArray,  # temp
]:
    """Evolves the ionization fraction over one timestep for the whole grid

    Warning: Calling this function assumes that the radiation tables have
    previously been copied to the GPU using photo_table_to_device()

    Parameters
    ----------
    Hz :
        Hubble constant.
    dt :
        Timestep in seconds.
    dr :
        Cell dimension in each direction in cm.
    R_max :
        Value of maximum comoving distance for photons from source (type 3 LLS
        in original C2Ray). This value is given in cell units, but doesn't need
        to be an integer.
    src_flux :
        Array containing the total ionizing flux of each source, normalized by
        S_star (1e48 by default).
    src_pos :
        Array containing the 3D grid position of each source, in Fortran
        indexing (from 1).
    temp :
        The initial temperature of each cell in K.
    ndens :
        The hydrogen number density of each cell in cm^-3.
    xh :
        The initial ionized fraction of each cell for each species (HII, HeII,
        HeIII).
    chems :
        Parameters used by the chemistry solver.
    logtau :
        Tuple of (start, step, num) describing the log tau axis of the photo
        tables.
    logtemp :
        Tuple of (start, step, num) describing the log temperature axis of the
        cooling tables.
    convergence_fraction :
        Which fraction of the cells can be left unconverged to improve
        performance (usually ~ 1e-4).
    use_mpi :
        Distribute batches of sources to other processes with MPI.
    rank :
        The MPI rank of this process.
    nprocs :
        The total number of MPI processes.

    Returns
    -------
    xh_int : tuple of float 3D-array
        The updated ionization fraction of each cell at the end of the timestep
        for each species (HII, HeII, HeIII).
    phion : tuple of float 3D-array
        Photo-ionization rate of each cell due to all sources for each species
        (HI, HeI, HeII).
    pheat : float 3D-array
        Photo-heating rate of each cell due to all sources for each species
        (HI, HeI, HeII).
    temp_int : float 3D-array
        Updated temperature of each cell at the end of the timestep.
    """
    if not is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(id)"
        )

    rank_prefix = f"[Rank={rank}] " if use_mpi else ""

    # Problem dimensions.
    N, _, _ = mesh_shape = ndens.shape
    num_cells = np.prod(mesh_shape)
    num_src, *_ = src_flux.shape

    # Convergence criteria.
    tot_xh = FractionStats((2.0 * num_cells,) * 6)
    conv_criterion = min(int(convergence_fraction * num_cells), (num_src - 1) / 3)
    converged = False

    logger.info(f"""Calling evolve3D...
dr [Mpc]: {dr / 3.086e24:.3e}
dt [years]: {dt / 3.15576e07:.3e}
Running on {num_src:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}
Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: HII = {xh[0].mean():.3e}, HeII = {xh[1].mean():.3e}, HeIII = {xh[2].mean():.3e},
Convergence Criterion (Number of points): {conv_criterion: n}
""")

    # Prepare source data for GPU. If using MPI, use a subset of sources for each rank.
    src_pos = src_pos.astype(np.int32)
    src_flux = src_flux.astype(np.float64)
    if use_mpi:
        chunk = distribute_jobs(num_src, nprocs, rank)

        # Overwrite number of sources.
        num_src = chunk.stop - chunk.start
        src_pos = src_pos[:, chunk]
        src_flux = src_flux[chunk]

        with allow_rank_logging(rank):
            logger.info(f"{rank_prefix}{num_src} sources.")

    # Copy positions & fluxes of sources to the GPU.
    assert libasora is not None
    libasora.source_data_to_device(src_pos.ravel(), src_flux.ravel())

    # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
    assert libasora is not None
    ndens = np.ravel(ndens).astype(np.float64)
    libasora.density_to_device(ndens)

    # Initialize average and intermediate results.
    xHII, xHeII, xHeIII = xh

    xHII = np.ravel(xHII).astype(np.float64)
    xHII_av = xHII.copy()
    xHII_int = xHII.copy()

    xHeII = np.ravel(xHeII).astype(np.float64)
    xHeII_av = xHeII.copy()
    xHeII_int = xHeII.copy()

    xHeIII = np.ravel(xHeIII).astype(np.float64)
    xHeIII_av = xHeIII.copy()
    xHeIII_int = xHeIII.copy()

    # Initialize ionization and heating rate arrays.
    phion_HI = np.empty_like(ndens)
    phion_HeI = np.empty_like(ndens)
    phion_HeII = np.empty_like(ndens)
    pheat = np.empty_like(ndens)

    rt = RadiationTables()
    sigmas = rt.cross_sections
    heat_factors = rt.factors
    nfreq = len(sigmas[0])

    # Prepare other inputs
    temp = np.ravel(temp).astype(np.float64)
    temp_int = temp.copy()
    clump = np.ravel(clump).astype(np.float64)

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
            R_max,
            *sigmas,
            heat_factors,
            nfreq,
            dr,
            xHII_av,
            xHeII_av,
            xHeIII_av,
            phion_HI,
            phion_HeI,
            phion_HeII,
            pheat,
            num_src,
            N,
            *logtau,
            src_batch_size,
        )

        time_end = time.perf_counter()
        with allow_rank_logging(rank):
            logger.info(
                f"{rank_prefix}...took {display_seconds(time_end - time_start)}"
            )

        if use_mpi:
            # Collect results from the different MPI processors
            comm.Allreduce(MPI.IN_PLACE, [phion_HI, MPI.DOUBLE], op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, [phion_HeI, MPI.DOUBLE], op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, [phion_HeII, MPI.DOUBLE], op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, [pheat, MPI.DOUBLE], op=MPI.SUM)

        if rank == 0:
            # ---------------------
            # (2): ODE Solving Step
            # ---------------------
            logger.info("Doing Chemistry...")

            time_start = time.perf_counter()

            # Apply the global rates to compute the updated ionization fraction
            # This function updates xh_av, xh_int and temp_int.
            conv_flag = libasora.chemistry_global_pass(
                dt,
                Hz,  # must get this from outside
                temp,
                temp_int,
                xHII,
                xHII_av,
                xHII_int,
                xHeII,
                xHeII_av,
                xHeII_int,
                xHeIII,
                xHeIII_av,
                xHeIII_int,
                phion_HI,
                phion_HeI,
                phion_HeII,
                pheat,
                clump,
                False,
                *logtemp,
            )

            time_end = time.perf_counter()
            logger.info(f"  took {display_seconds(time_end - time_start)}")

            # ----------------------------
            # (3): Test Global Convergence
            # ----------------------------

            tot_xh_new = FractionStats.from_xh(xHII_int, xHeII_int, xHeIII_int)
            rel_change = tot_xh.relative_change(tot_xh_new)

            logger.info(
                f"Number of non-converged points: {conv_flag} of {num_cells} ({conv_flag / num_cells:.3%}), "
                f"Relative change in: HII ionfrac {rel_change.HII[0]:.2e}, "
                f"HeII ionfrac {rel_change.HeII[0]:.2e}, HeIII ionfrac {rel_change.HeIII[0]:.2e}"
            )

            converged = (conv_flag < conv_criterion) or all(
                xh < convergence_fraction for xh in rel_change.data
            )
            n_count += 1
            tot_xh = tot_xh_new

        if use_mpi:
            # Broadcast ionised fraction field
            MPI.COMM_WORLD.Bcast([xHII_av, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xHeII_av, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xHeIII_av, MPI.DOUBLE], root=0)

            # Broadcast convergence
            converged = MPI.COMM_WORLD.bcast(converged, root=0)

    if use_mpi:
        MPI.COMM_WORLD.Bcast([xHII_int, MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([xHeII_int, MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([xHeIII_int, MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([temp_int, MPI.DOUBLE], root=0)

    if rank == 0:
        logger.info(
            f"Multiple source convergence reached after {n_count} ray-tracing iterations."
        )

    return (
        (
            xHII_int.reshape(mesh_shape),
            xHeII_int.reshape(mesh_shape),
            xHeIII_int.reshape(mesh_shape),
        ),
        (
            phion_HI.reshape(mesh_shape),
            phion_HeI.reshape(mesh_shape),
            phion_HeII.reshape(mesh_shape),
        ),
        pheat.reshape(mesh_shape),
        temp_int.reshape(mesh_shape),
    )


def _evolve3D_c2ray(
    Hz: float,
    dt: float,
    dr: float,
    R_max: float,
    src_flux: FloatArray,
    src_pos: IntArray,
    src_batch_size: int,
    max_subbox: int,
    subboxsize: int,
    loss_fraction: float,
    temp: FloatArray,
    ndens: FloatArray,
    clump: FloatArray,
    xh: tuple[FloatArray, FloatArray, FloatArray],
    photo_thin_table: FloatArray,
    photo_thick_table: FloatArray,
    convergence_fraction: float,
    sigma: float,
    chems: ChemistryParams,
    logtau: tuple[float, float, int],
    use_mpi: bool,
    rank: int,
    nprocs: int,
    *args,
    **kwargs,
) -> tuple[
    tuple[FloatArray, FloatArray, FloatArray],  # xfrac
    tuple[FloatArray, FloatArray, FloatArray],  # ion rate
    FloatArray,  # heat rate
    FloatArray,  # temp
]:
    """Evolves the ionization fraction over one timestep for the whole grid

    Parameters
    ----------
    Hz :
        Hubble constant.
    dt :
        Timestep in seconds.
    dr :
        Cell dimension in each direction in cm.
    R_max :
        Value of maximum comoving distance for photons from source (type 3 LLS
        in original C2Ray). This value is given in cell units, but doesn't need
        to be an integer.
    src_flux :
        Array containing the total ionizing flux of each source, normalized by
        S_star (1e48 by default).
    src_pos :
        Array containing the 3D grid position of each source, in Fortran
        indexing (from 1).
    max_subbox :
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no
        effect when use_gpu is true.
    subboxsize :
        ...
    loss_fraction :
        Fraction of remaining photons below we stop ray-tracing (subbox
        technique). Has no effect when use_gpu is true.
    temp :
        The initial temperature of each cell in K.
    ndens :
        The hydrogen number density of each cell in cm^-3.
    xh :
        The initial ionized fraction of each cell.
    clump :
        The clumping factor of each cell.
    photo_thin_table :
    photo_thick_table :
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this
        table needs to have been copied to the GPU in a separate (previous)
        step, using photo_table_to_device().
    convergence_fraction :
        Which fraction of the cells can be left unconverged to improve
        performance (usually ~ 1e-4).
    sigma :
        Constant photoionization cross-section of hydrogen in cm^2.
    chems :
        Parameters used by the chemistry solver.
    logtau :
        Tuple of (start, step, num) describing the log tau axis of the photo
        tables.
    use_mpi :
        Distribute batches of sources to other processes with MPI.
    rank :
        The MPI rank of this process.
    nprocs :
        The total number of MPI processes.

    Returns
    -------
    xh_int : 3D-array of dtype float
        The updated ionization fraction of each cell at the end of the
        timestep.
    phion : 3D-array of dtype float
        Photoionization rate of each cell due to all sources.
    pheat: 3D-array of dtype float
        Photoheating rate of each cell due to all sources.
    temp_int : 3D-array of dtype float
        Updated temperature of each cell at the end of the timestep.
    """
    rank_prefix = f"Rank={rank}: " if use_mpi else ""

    # Problem dimensions.
    N, _, _ = mesh_shape = ndens.shape
    num_cells = np.prod(mesh_shape)
    num_src, *_ = src_flux.shape

    # Convergence criteria.
    tot_xh = FractionStats((2.0 * num_cells,) * 6)
    conv_criterion = min(int(convergence_fraction * num_cells), (num_src - 1) / 3)
    converged = False

    logger.info(f"""Calling evolve3D...
dr [Mpc]: {dr / 3.086e24:.3e}
dt [years]: {dt / 3.15576e07:.3e}
Running on {num_src:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}
Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: HII = {xh[0].mean():.3e}, HeII = {xh[1].mean():.3e}, HeIII = {xh[2].mean():.3e},
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
    xHII, xHeII, xHeIII = xh

    xHII = np.ravel(xHII).astype(np.float64, order="F")
    xHII_av = xHII.copy()
    xHII_int = xHII.copy()

    xHeII = np.ravel(xHeII).astype(np.float64, order="F")
    xHeII_av = xHeII.copy()
    xHeII_int = xHeII.copy()

    xHeIII = np.ravel(xHeIII).astype(np.float64, order="F")
    xHeIII_av = xHeIII.copy()
    xHeIII_int = xHeIII.copy()

    # Initialize rate arrays.
    phion_HI = np.zeros(mesh_shape, dtype=np.float64, order="F")
    phion_HeI = np.zeros_like(phion_HI)
    phion_HeII = np.zeros_like(phion_HI)
    pheat_HI = np.zeros_like(phion_HI)
    pheat_HeI = np.zeros_like(phion_HI)
    pheat_HeII = np.zeros_like(phion_HI)
    coldensh_out = np.zeros_like(phion_HI)

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
            xHII_av,
            phion_HI,
            pheat_HI,
            loss_fraction,
            photo_thin_table,
            photo_thick_table,
            heat_thin_table,
            heat_thick_table,
            logtau[0],
            logtau[1],
            R_max,
        )

        time_end = time.perf_counter()
        with allow_rank_logging(rank):
            logger.info(f"  took {display_seconds(time_end - time_start)}")

        logger.info(
            f"Average number of subboxes: {nsubbox / num_src:n}, Total photon loss: {photonloss:.3e}"
        )

        if use_mpi:
            # Collect results from the different MPI processors
            comm.Allreduce(MPI.IN_PLACE, [phion_HI, MPI.DOUBLE], op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, [pheat_HI, MPI.DOUBLE], op=MPI.SUM)

        if rank == 0:
            # ---------------------
            # (2): ODE Solving Step
            # ---------------------
            logger.info("Doing Chemistry...")

            time_start = time.perf_counter()

            # Apply the global rates to compute the updated ionization fraction
            conv_flag = libc2ray.chemistry.global_pass(
                dt,
                Hz,
                ndens,
                temp,
                np.zeros_like(temp),
                xHII,
                xHII_av,
                xHII_int,
                xHeII,
                xHeII_av,
                xHeII_int,
                xHeIII,
                xHeIII_av,
                xHeIII_int,
                phion_HI,
                phion_HeI,
                phion_HeII,
                pheat_HI,
                pheat_HeI,
                pheat_HeII,
                clump,
                True,
            )

            time_end = time.perf_counter()
            logger.info(f"...took {display_seconds(time_end - time_start)}")

            # ----------------------------
            # (3): Test Global Convergence
            # ----------------------------
            tot_xh_new = FractionStats.from_xh(xHII_int, xHeII_int, xHeIII_int)
            rel_change = tot_xh.relative_change(tot_xh_new)

            logger.info(
                f"Number of non-converged points: {conv_flag} of {num_cells} ({conv_flag / num_cells:.3%}), "
                f"Relative change in: HII ionfrac {rel_change.HII[0]:.2e}, "
                f"HeII ionfrac {rel_change.HeII[0]:.2e}, HeIII ionfrac {rel_change.HeIII[0]:.2e}"
            )

            converged = (conv_flag < conv_criterion) or all(
                xh < convergence_fraction for xh in rel_change.data
            )
            n_count += 1
            tot_xh = tot_xh_new

        if use_mpi:
            # Broadcast ionised fraction field
            MPI.COMM_WORLD.Bcast([xHII_av, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xHeII_av, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xHeIII_av, MPI.DOUBLE], root=0)

            # Broadcast convergence
            converged = MPI.COMM_WORLD.bcast(converged, root=0)

    if use_mpi:
        MPI.COMM_WORLD.Bcast([xHII_int, MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([xHeII_int, MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([xHeIII_int, MPI.DOUBLE], root=0)

    if rank == 0:
        logger.info(
            f"Multiple source convergence reached after {n_count} ray-tracing iterations."
        )

    return (
        (xHII_int, xHeII_int, xHeIII_int),
        (phion_HI, phion_HeI, phion_HeII),
        pheat_HI,
        temp,
    )


def evolve3D(
    **kwargs,
) -> tuple[
    tuple[FloatArray, FloatArray, FloatArray],  # xfrac
    tuple[FloatArray, FloatArray, FloatArray],  # ion rate
    FloatArray,  # heat rate
    FloatArray,  # temp
]:
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
