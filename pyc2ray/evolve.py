import array
import logging
import time

import numpy as np
from mpi4py import MPI

from .asora_core import is_device_init
from .load_extensions import libasora, libc2ray
from .utils import display_time
from .utils.logutils import disable_newline
from .utils.sourceutils import FloatArray, IntArray, format_sources
from .domain.domain_decomposition_utils import Group, Source, Grid, build_groups, assign_groups_to_ranks, log_domain_decomposition_assignments, evaluate_group

__all__ = ["evolve3D"]

logger = logging.getLogger(__name__)

# =========================================================================
# This file contains the main time-evolution subroutine, which updates
# the ionization state of the whole grid over one timestep, using the
# C2Ray method.
#
# The raytracing step can use either the sequential (subbox, cubic)
# technique which runs in Fortran on the CPU or the accelerated technique,
# which runs using the ASORA library on the GPU.
#
# When using the latter, some notes apply:
# For performance reasons, the program minimizes the frequency at which
# data is moved between the CPU and the GPU (this is a big bottleneck).
# In particular, the radiation tables, which in principle shouldn't change
# over the run of a simulation, need to be copied separately to the GPU
# using the photo_table_to_device() method of the module. This is done
# automatically when using the C2Ray subclasses but must be done manually
# if for some reason you are calling the evolve3D routine directly without
# using the C2Ray subclasses.
#
# This file defines two variants of evolve3D: The reference, single-gpu
# version, and a MPI version which enables usage on multiple GPU nodes.
# =========================================================================


def evolve3D(
    dt: float,
    dr: float,
    src_flux: FloatArray,
    src_pos: IntArray,
    src_batch_size: int,
    activate_domain_decomposition: bool,
    use_gpu: bool,
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
    sig: float,
    bh00: float,
    albpow: float,
    colh0: float,
    temph0: float,
    abu_c: float,
) -> tuple[FloatArray, FloatArray]:
    """Evolves the ionization fraction over one timestep for the whole grid

    Warning: Calling this function with use_gpu = True assumes that the radiation
    tables have previously been copied to the GPU using photo_table_to_device()

    Parameters
    ----------
    dt : float
        Timestep in seconds
    dr : float
        Cell dimension in each direction in cm
    src_flux : 1D-array of shape (numsrc)
        Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default)
    src_pos : 2D-array of shape (3,numsrc)
        Array containing the 3D grid position of each source, in Fortran indexing (from 1)
    src_batch_size : int
        Number of sources to process in each batch
    activate_domain_decomposition : bool
        Whether or not compute evolution by using source grouping and domain decomposition.
    use_gpu : bool
        Whether or not to use the GPU-accelerated ASORA library for raytracing.
    max_subbox : int
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true
    subboxsize : int
        ...
    loss_fraction : float
        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true
    temp : 3D-array
        The initial temperature of each cell in K
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    xh : 3D-array
        The initial ionized fraction of each cell
    photo_thin_table : 1D-array
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
        in a separate (previous) step, using photo_table_to_device()
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    dlogtau : float
        Step size of the logτ-table
    R_max_LLS : float
        Value of maximum comoving distance for photons from source (type 3 LLS in original C2Ray). This value is
        given in cell units, but doesn't need to be an integer
    convergence_fraction : float
        Which fraction of the cells can be left unconverged to improve performance (usually ~ 1e-4)
    sig : float
        Constant photoionization cross-section of hydrogen in cm^2. TODO: replace by general (frequency-dependent)
        case.
    bh00 : float
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation
    albpow : float
        Power-law index for the H recombination parameter
    colh0 : float
        Hydrogen collisional ionization parameter
    temph0 : float
        Hydrogen ionization energy expressed in K
    abu_c : float
        Carbon abundance

    Returns
    -------
    xh_new : 3D-array
        The updated ionization fraction of each cell at the end of the timestep
    phi_ion : 3D-array
        Photoionization rate of each cell due to all sources
    """

    # Allow a call with GPU only if
    # 1. the asora library is present and
    # 2. the GPU memory has been allocated using device_init()
    if use_gpu and not is_device_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )

    # Set some constant sizes
    NumSrc = src_flux.shape[0]  # Number of sources
    N = temp.shape[0]  # Mesh size
    NumCells = N * N * N  # Number of cells/points
    conv_flag = NumCells  # Flag that counts the number of non-converged cells (initialized to non-convergence)
    NumTau = photo_thin_table.shape[0]

    # Convergence Criteria
    conv_criterion = min(int(convergence_fraction * NumCells), (NumSrc - 1) / 3)

    # Initialize convergence metrics
    prev_sum_xh1_int: float = 2 * NumCells
    prev_sum_xh0_int: float = 2 * NumCells
    converged = False
    if rank != 0:
        xh_new = np.empty_like(xh)
    niter = 0

    # initialize average and intermediate results to values at beginning of timestep
    xh_av = np.copy(xh)
    xh_intermed = np.copy(xh)

    # Run source grouping and domain decomposition
    is_domain_decomposition_active = use_mpi and use_gpu and activate_domain_decomposition
    if is_domain_decomposition_active:

        # Only do source grouping on rank 0, then broadcast the result to the other ranks
        ranks_groups = None
        ranks_costs = 0.0
        if rank == 0:
            # TODO CB: avoid conversion into physical units and back.
            logger.info(f"Running on {nprocs} MPI ranks, doing source grouping and domain decomposition...")
            # TODO CB: check radius value, I'm currently reducing it for testing purposes.
            source_groups = build_groups(
                sources=[Source(i, pos=(np.array(src_pos[:, i], dtype=float) - 0.5) * dr,
                                strength=src_flux[i], radius=R_max_LLS*dr) for i in range(NumSrc)],
                grid=Grid(num_cells=N, dx=dr),
                nsrc_max = 1)
            logger.info(f"Created {len(source_groups)} source groups for domain decomposition.")
            ranks_groups, ranks_costs = assign_groups_to_ranks(source_groups, nranks=nprocs)

            # Rank-0 inspection of assignment results before scatter.
            log_domain_decomposition_assignments(
                ranks_groups=ranks_groups,
                ranks_costs=ranks_costs,
                dr=dr,
            )

        # Broadcast source groups to other ranks
        # TODO CB: barrier not needed
        MPI.COMM_WORLD.Barrier() # make sure rank 0 has finished building the groups before other ranks try to receive them
        local_groups = MPI.COMM_WORLD.scatter(ranks_groups, root=0)

        local_group = None
        if len(local_groups) == 1:
            local_group = local_groups[0]
        elif len(local_groups) > 1:
            logger.error(f"Rank {rank} received more than one group, which is not currently supported.")
            raise NotImplementedError("Multiple groups per rank not supported yet.")
        else:
            logger.error(f"Rank {rank} did not receive any group, which is not currently supported.")
            raise NotImplementedError("No group assigned to rank, which is not supported yet.")

    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
    if use_gpu:
        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
        assert libasora is not None
        is_periodic_mode_active = bool(libasora.is_periodic_mode_active())
        print(f"Rank {rank} is_periodic_mode_active: {is_periodic_mode_active}.", flush=True)
        # If domain_decomposition is active we can limit the grid to be copied to the GPU to the one overlapping with the local groups.
        if is_domain_decomposition_active:
            if local_group is not None:
                # Get total number of cells per side for the full bounding box of the local group, including the part outside the grid domain.
                # TODO CB: if we don't want to modify ASORA code we need to include out-of-domain cells in the subdomain and fill them with zeros
                sub_mesh_size = local_group.get_full_num_cells_per_side()

                # local_group.cells = (full_min, full_max, clipped_min, clipped_max, volume), where max indexes are inclusive.
                full_min = np.asarray(local_group.cells[0], dtype=int)

                xh_local = np.empty((sub_mesh_size, sub_mesh_size, sub_mesh_size), dtype=np.float64)
                ndens_local = np.empty((sub_mesh_size, sub_mesh_size, sub_mesh_size), dtype=np.float64)

                if is_periodic_mode_active:

                    # Build periodic index vectors for the full local cube. This handles
                    # cells outside the global box by wrapping to the opposite side.
                    gi = (np.arange(sub_mesh_size, dtype=np.int64) + full_min[0]) % N
                    gj = (np.arange(sub_mesh_size, dtype=np.int64) + full_min[1]) % N
                    gk = (np.arange(sub_mesh_size, dtype=np.int64) + full_min[2]) % N

                    # Optimize memory layout
                    xh_local = np.ascontiguousarray(
                        xh[np.ix_(gi, gj, gk)], dtype=np.float64
                    )
                    ndens_local = np.ascontiguousarray(
                        ndens[np.ix_(gi, gj, gk)], dtype=np.float64
                    )

                else:

                    # TODO CB: unify with case above
                    clipped_min = np.asarray(local_group.cells[2], dtype=int)
                    clipped_max = np.asarray(local_group.cells[3], dtype=int)

                    local_offset = clipped_min - full_min
                    clipped_shape = clipped_max - clipped_min + 1

                    xh_local.fill(-1.0)
                    ndens_local.fill(-1.0)

                    xh_local[
                        local_offset[0]:local_offset[0] + clipped_shape[0],
                        local_offset[1]:local_offset[1] + clipped_shape[1],
                        local_offset[2]:local_offset[2] + clipped_shape[2],
                    ] = xh[
                        clipped_min[0]:clipped_max[0] + 1,
                        clipped_min[1]:clipped_max[1] + 1,
                        clipped_min[2]:clipped_max[2] + 1,
                    ]

                    ndens_local[
                        local_offset[0]:local_offset[0] + clipped_shape[0],
                        local_offset[1]:local_offset[1] + clipped_shape[1],
                        local_offset[2]:local_offset[2] + clipped_shape[2],
                    ] = ndens[
                        clipped_min[0]:clipped_max[0] + 1,
                        clipped_min[1]:clipped_max[1] + 1,
                        clipped_min[2]:clipped_max[2] + 1,
                    ]

                xh_av_flat = np.ravel(xh_local).astype("float64", copy=True)
                ndens_flat = np.ravel(ndens_local).astype("float64", copy=True)
            else:
                # TODO CB: implement missing handling with no local group
                raise NotImplementedError(f"No group assigned to rank {rank}, which is not supported yet.")

        else:
            # Format input data for the CUDA extension module (flat arrays, C-types,etc)
            xh_av_flat = np.ravel(xh).astype("float64", copy=True)
            ndens_flat = np.ravel(ndens).astype("float64", copy=True)

        if use_mpi:
            if is_domain_decomposition_active:
                if local_group is not None:
                    local_source_ids = local_group.get_source_ids()
                    NumSrc = len(local_group.sources)
                    srcpos_flat, normflux_flat = format_sources(
                        # Source position in subdomain already respects ASORA convention for Fortran indexing
                        src_pos[:, local_source_ids] - full_min[:, None],
                        src_flux[local_source_ids]
                    )
                else:
                    raise NotImplementedError(f"No group assigned to rank {rank}, which is not supported yet.")

            else:
                # TODO:       #if(NumSrc > nprocs):
                perrank = NumSrc // nprocs
                i_start = int(rank * perrank)
                if rank != nprocs - 1:
                    i_end = int((rank + 1) * perrank)
                else:
                    i_end = NumSrc

                # overwrite number of sources
                NumSrc = i_end - i_start
                srcpos_flat, normflux_flat = format_sources(
                    src_pos[:, i_start:i_end], src_flux[i_start:i_end]
                )
                logger.info(f"...rank={rank:n} has {NumSrc:n} sources.")
        else:
            srcpos_flat, normflux_flat = format_sources(src_pos, src_flux)

        # Copy positions & fluxes of sources to the GPU in advance
        # TODO CB: in principle, we don't need to copy the sources in every timestep
        # since they don't change position or strength, right?
        MPI.COMM_WORLD.Barrier()
        libasora.source_data_to_device(srcpos_flat, normflux_flat)

        # Initialize Flat Column density & ionization rate arrays.
        # These are used to store the output of the raytracing module.
        # TODO CB: find a way to save memory
        if is_domain_decomposition_active:
            sub_phi_ion_flat = np.ravel(np.zeros((sub_mesh_size, sub_mesh_size, sub_mesh_size), dtype="float64"))
        phi_ion_flat = np.ravel(np.zeros((N, N, N), dtype="float64"))

        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
        assert libasora is not None
        libasora.density_to_device(ndens_flat)
        if use_mpi:
            logger.info("Copied source data to device.")
        else:
            logger.info(f"Rank {rank} copied source data to device.")

    # -----------------------------------------------------------
    # Start Evolve step, Iterate until convergence in <x> and <y>
    # -----------------------------------------------------------
    if rank == 0:
        n_count = 0

    logger.info(f"""Calling evolve3D...
dr [Mpc]: {dr / 3.086e24:.3e}
dt [years]: {dt / 3.15576e07:.3e}
Running on {NumSrc:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}
Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: {xh.mean():.3e}
Convergence Criterion (Number of points): {conv_criterion: n}
""")

    while not converged:
        niter += 1

        # --------------------
        # (1): Raytracing Step
        # --------------------
        trt0 = time.time()
        with disable_newline():
            if use_mpi:
                logger.info("Doing Raytracing...")
            else:
                logger.info(f"Rank={rank} is doing Raytracing...")

        # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
        if use_gpu:
            # Use GPU raytracing
            assert libasora is not None
            if is_domain_decomposition_active:

                # TODO CB: we can probably avoid this reshape and ravel if we are careful with the indexing in the CUDA kernel, but for now this is simpler to implement.
                # If this is not first iteration then we need to find subdomain xh_av_flat from the global one received from rank 0 after the broadcast. 
                # If this is the first iteration, xh_av_flat is already correctly initialized to the subdomain values.
                if niter > 1:
                    tmp_xh_av = np.reshape(xh_av_flat, (N, N, N))
                    if is_periodic_mode_active:
                        xh_local = np.ascontiguousarray(
                            tmp_xh_av[np.ix_(gi, gj, gk)], dtype=np.float64
                        )
                    else:
                        xh_local = np.empty((sub_mesh_size, sub_mesh_size, sub_mesh_size), dtype=np.float64)
                        xh_local.fill(-1.0)
                        xh_local[
                            local_offset[0]:local_offset[0] + clipped_shape[0],
                            local_offset[1]:local_offset[1] + clipped_shape[1],
                            local_offset[2]:local_offset[2] + clipped_shape[2],
                        ] = tmp_xh_av[
                            clipped_min[0]:clipped_max[0] + 1,
                            clipped_min[1]:clipped_max[1] + 1,
                            clipped_min[2]:clipped_max[2] + 1,
                        ]
                    xh_av_flat = np.ravel(xh_local).astype("float64", copy=True)
                    MPI.COMM_WORLD.Barrier()

                # TODO CB: avoid call duplication here.
                MPI.COMM_WORLD.Barrier()
                libasora.do_all_sources(
                    R_max_LLS,
                    sig,
                    dr,
                    xh_av_flat,
                    sub_phi_ion_flat,
                    NumSrc,
                    sub_mesh_size,
                    minlogtau,
                    dlogtau,
                    NumTau,
                    src_batch_size, # Determines the CUDA kernel grid size
                )
                MPI.COMM_WORLD.Barrier()
            else:
                libasora.do_all_sources(
                    R_max_LLS,
                    sig,
                    dr,
                    xh_av_flat,
                    phi_ion_flat,
                    NumSrc,
                    N,
                    minlogtau,
                    dlogtau,
                    NumTau,
                    src_batch_size, # Determines the CUDA kernel grid size
                )
        else:
            # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
            phi_ion = np.zeros((N, N, N), order="F")
            # So far in evolve we ignore heating (not considered in chemistry),
            # but the raytracing function requires heating tables as argument
            phi_heat = np.zeros((N, N, N), order="F")
            coldensh_out = np.zeros((N, N, N), order="F")
            # Use CPU raytracing with subbox optimization
            nsubbox, photonloss = libc2ray.raytracing.do_all_sources(
                src_flux,
                src_pos,
                max_subbox,
                subboxsize,
                coldensh_out,
                sig,
                dr,
                ndens,
                xh_av,
                phi_ion,
                phi_heat,
                loss_fraction,
                photo_thin_table,
                photo_thick_table,
                np.zeros(NumTau),
                np.zeros(NumTau),  # Eventually we'll add heating tables here
                minlogtau,
                dlogtau,
                R_max_LLS,
            )

        trt1 = time.time() - trt0
        if use_mpi:
            logger.info(f"  rank={rank} took {display_time(trt1)}.")
        else:
            logger.info(f"  took {display_time(trt1)}")

        # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
        if use_gpu:
            if is_domain_decomposition_active:
                # TODO CB: find a cleaner way to avoid this fill: the DD code doesn't use
                # phi_ion_flat directly so this clean up step is unclear (but necessary)
                phi_ion_flat.fill(0.0) # make sure to reset the global phi_ion array before copying the subdomain result into it, since we will do a sum reduction across ranks later
            phi_ion = np.reshape(phi_ion_flat, (N, N, N))
        else:
            logger.info(
                f"Average number of subboxes: {nsubbox / NumSrc:n}, Total photon loss: {photonloss:.3e}"
            )

        if use_mpi:

            # Copy the subbox result to the full phi_ion array, taking into account the position of the local group in the full grid.
            if is_domain_decomposition_active:
                sub_phi_ion = np.reshape(sub_phi_ion_flat, (sub_mesh_size, sub_mesh_size, sub_mesh_size))

                if is_periodic_mode_active:
                    # In periodic mode, the local group can wrap around the edges of the global grid, 
                    # so we need to use the periodic index vectors to copy the data to the correct 
                    # positions in the global phi_ion array.
                    # TODO CB: optimize
                    for i_local in range(sub_mesh_size):
                        for j_local in range(sub_mesh_size):
                            for k_local in range(sub_mesh_size):
                                i_global = (full_min[0] + i_local) % N
                                j_global = (full_min[1] + j_local) % N
                                k_global = (full_min[2] + k_local) % N
                                # TODO CB:
                                # += is needed when more groups are assigned to the same rank or 
                                # if subdomain is larger than the global domain (which should not happen in practice)
                                phi_ion[i_global, j_global, k_global] += sub_phi_ion[i_local, j_local, k_local]
                else:
                    phi_ion[
                        clipped_min[0]:clipped_max[0] + 1,
                        clipped_min[1]:clipped_max[1] + 1,
                        clipped_min[2]:clipped_max[2] + 1,
                    ] = sub_phi_ion[
                        local_offset[0]:local_offset[0] + clipped_shape[0],
                        local_offset[1]:local_offset[1] + clipped_shape[1],
                        local_offset[2]:local_offset[2] + clipped_shape[2],
                    ]
            MPI.COMM_WORLD.Barrier()
            # Collect results from the different MPI processors
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [phi_ion, MPI.DOUBLE], op=MPI.SUM)

        if rank == 0:
            # ---------------------
            # (2): ODE Solving Step
            # ---------------------
            tch0 = time.time()
            with disable_newline():
                logger.info("Doing Chemistry...")
            # Apply the global rates to compute the updated ionization fraction
            conv_flag = libc2ray.chemistry.global_pass(
                dt,
                ndens,
                temp,
                xh,
                xh_av,
                xh_intermed,
                phi_ion,
                clump,
                bh00,
                albpow,
                colh0,
                temph0,
                abu_c,
            )

            # TODO: the line below is the same function but completely in python
            # (much slower then the fortran version, due to a lot of loops)
            # xh_intermed, xh_av, conv_flag = global_pass(
            #     dt, ndens, temp, xh, xh_av, xh_intermed, phi_ion,
            #     clump, bh00, albpow, colh0, temph0, abu_c,
            # )

            logger.info(f"  took {(time.time() - tch0): .1f} s.")

            # ----------------------------
            # (3): Test Global Convergence
            # ----------------------------
            sum_xh1_int = np.sum(xh_intermed)
            sum_xh0_int = np.sum(1.0 - xh_intermed)

            if sum_xh1_int > 0.0:
                rel_change_xh1 = np.abs((sum_xh1_int - prev_sum_xh1_int) / sum_xh1_int)
            else:
                rel_change_xh1 = 1.0

            if sum_xh0_int > 0.0:
                rel_change_xh0 = np.abs((sum_xh0_int - prev_sum_xh0_int) / sum_xh0_int)
            else:
                rel_change_xh0 = 1.0

            # Display convergence
            logger.info(
                f"Number of non-converged points: {conv_flag} of {NumCells} ({conv_flag / NumCells * 100: .3f} % ), "
                f"Relative change in ionfrac: {rel_change_xh1: .2e}",
            )
            converged = (conv_flag < conv_criterion) or (
                (rel_change_xh1 < convergence_fraction)
                and (rel_change_xh0 < convergence_fraction)
            )
            # converged = True # TODO CB: re-enable convergence check after testing
            # increase the convergence iteration counter
            n_count += 1

            # Set previous metrics to current ones and repeat if not converged
            prev_sum_xh1_int = sum_xh1_int
            prev_sum_xh0_int = sum_xh0_int

            # Finally, when using GPU, need to reshape xh back for the next ASORA call
            if use_gpu and (not converged or is_domain_decomposition_active):
                xh_av_flat = np.ravel(xh_av)

        if use_mpi:
            # broadcast ionised fraction field
            if is_domain_decomposition_active and rank != 0:
                # Collective ops require equal buffer sizes on all ranks.
                xh_av_flat = np.empty(N * N * N, dtype=np.float64)

            # TODO CB: barrier not needed
            MPI.COMM_WORLD.Barrier() # make sure all ranks have reached this point before broadcasting

            MPI.COMM_WORLD.Bcast([xh_av_flat, MPI.DOUBLE], root=0)
            MPI.COMM_WORLD.Bcast([xh_intermed, MPI.DOUBLE], root=0)

            # convert the bool variable to bit
            # converged_array = array.array("i", [converged])
            converged_array = array.array("i", [int(converged)])

            # braodcast convergence to the other ranks
            MPI.COMM_WORLD.Bcast(converged_array, root=0)
            if rank != 0:
                converged = bool(converged_array[0])

    if rank == 0:
        # When converged, return the updated ionization fractions at the end of the timestep
        logger.info(
            f"Multiple source convergence reached after {n_count} ray-tracing iterations."
        )
        xh_new = xh_intermed

    if use_mpi:
        # braodcast final result
        MPI.COMM_WORLD.Bcast([xh_new, MPI.DOUBLE], root=0)

    return xh_new, phi_ion
