from unittest.mock import Mock, patch

import astropy.constants as cst
import astropy.units as u
import numpy as np
import pytest
from mpi4py import MPI

from pyc2ray.domain.domain_decomposition_handler import DomainDecompositionHandler
from pyc2ray.evolve import ChemistryParams, evolve3D
from pyc2ray.parameters import DomainDecompositionParameters
from pyc2ray.radiation.blackbody import BlackBodySource
from pyc2ray.radiation.common import make_tau_table


@pytest.fixture
def mock_c2ray():
    with patch("pyc2ray.evolve.libc2ray") as mock:
        mock.configure_mock(
            **{
                "chemistry.global_pass": Mock(return_value=1),
                "raytracing.do_all_sources": Mock(return_value=(10, 0.1)),
            }
        )
        yield mock


@pytest.fixture
def mock_asora():
    with (
        patch("pyc2ray.evolve.is_device_init", return_value=True),
        patch("pyc2ray.evolve.libasora") as mock,
    ):
        mock.configure_mock(chemistry_global_pass=Mock(return_value=1))
        yield mock


@pytest.fixture
def mock_asora_domain_decomposition():
    """Mock the GPU-side helpers used by the domain decomposition path.

    The domain decomposition module itself (grid, source grouping, subdomain
    mapping) runs for real on the CPU; only the ASORA/GPU calls are mocked.
    """
    with (
        patch("pyc2ray.evolve.is_device_init", return_value=True),
        patch("pyc2ray.evolve.prepare_grid_buffers"),
        patch("pyc2ray.evolve.libasora") as mock,
    ):
        mock.configure_mock(chemistry_global_pass=Mock(return_value=1))
        yield mock


def call_evolve3D(
    use_gpu: bool = False,
    rank: int = 0,
    domain_decomposition_params: DomainDecompositionParameters | None = None,
    nprocs: int = 8,
    fields: dict[str, np.ndarray] | None = None,
):
    N = 32

    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, N, size=(3, 10), dtype=np.int32)
    src_flux = rng.uniform(1e10, 1e14, size=10).astype(np.float64)
    src_flux *= 1e-46

    # The domain decomposition path builds a per-group local grid sized by the
    # sources' radius of influence (R_max_LLS cells). On this small test grid a
    # radius of 15 cells would make local grids larger than the global grid,
    # which the periodic mapping forbids. Use a smaller radius in that case.
    if domain_decomposition_params is not None:
        R_max_LLS = 2.0
    else:
        R_max_LLS = 15.0

    shape = (N, N, N)
    ndens = np.empty(shape, order="F")
    xh = np.full(shape, 1.2e-3, order="F")
    temp = np.full(shape, 1e4, order="F")
    clump = np.full(shape, 1.0, dtype=np.float64, order="F")
    if fields is not None:
        ndens, xh, temp, clump = (fields[k] for k in ("ndens", "xh", "temp", "clump"))

    minlogtau, maxlogtau, num_tau = -20.0, 4.0, 20000
    tau, dlogtau = make_tau_table(minlogtau, maxlogtau, num_tau)
    freq_min, freq_max = (
        (13.598 * u.eV / cst.h).to("Hz").value,
        (54.416 * u.eV / cst.h).to("Hz").value,
    )
    sigma = 6.30e-18
    radsource = BlackBodySource(1e5, False, freq_min, sigma)
    photo_thin_table, photo_thick_table = radsource.make_photo_table(
        tau, freq_min, freq_max, 1e48
    )

    colh0 = 1.3e-8 * 0.83 * 1.0 / 13.598**2
    temph0 = 13.598 / (cst.k_B * u.K).to("eV").value

    dr = (1 * u.Mpc).cgs.value / N

    # The domain decomposition path consumes a decomposition handler owned by the caller.
    decomposition = None
    if domain_decomposition_params is not None:
        decomposition = DomainDecompositionHandler(MPI.COMM_WORLD)
        decomposition.update_decomposition(
            cell_size=dr,
            src_pos=src_pos.T,
            src_flux=src_flux,
            N=N,
            R_max_LLS=R_max_LLS,
            src_batch_size=8,
            num_tau=num_tau,
            is_domain_periodic=True,
            domain_decomposition_params=domain_decomposition_params,
        )

    return evolve3D(
        dt=1e3,
        dr=dr,
        src_flux=src_flux,
        src_pos=src_pos,
        src_batch_size=8,
        use_gpu=use_gpu,
        max_subbox=1000,
        subboxsize=128,
        loss_fraction=1e-2,
        use_mpi=False,
        rank=rank,
        nprocs=nprocs,
        temp=temp,
        ndens=ndens,
        xh=xh,
        clump=clump,
        photo_thin_table=photo_thin_table,
        photo_thick_table=photo_thick_table,
        minlogtau=minlogtau,
        dlogtau=dlogtau,
        R_max_LLS=R_max_LLS,
        convergence_fraction=1e-4,
        sigma=sigma,
        chems=ChemistryParams(2.59e-13, -0.7, colh0, temph0, 7.1e-7),
        decomposition=decomposition,
    )


def test_evolve3D_no_gpu_root_rank(mock_c2ray, mock_asora):
    call_evolve3D(use_gpu=False, rank=0)

    mock_asora.source_data_to_device.assert_not_called()
    mock_asora.density_to_device.assert_not_called()
    mock_asora.do_all_sources.assert_not_called()
    mock_asora.chemistry_global_pass.assert_not_called()

    mock_c2ray.chemistry.global_pass.assert_called()
    mock_c2ray.raytracing.do_all_sources.assert_called()


def test_evolve3D_yes_gpu_root_rank(mock_c2ray, mock_asora):
    call_evolve3D(use_gpu=True, rank=0)

    mock_asora.source_data_to_device.assert_called()
    mock_asora.density_to_device.assert_called()
    mock_asora.do_all_sources.assert_called()
    mock_asora.chemistry_global_pass.assert_called()

    mock_c2ray.chemistry.global_pass.assert_not_called()
    mock_c2ray.raytracing.do_all_sources.assert_not_called()


def test_evolve3D_yes_gpu_domain_decomposition_root_rank(
    mock_c2ray, mock_asora_domain_decomposition
):
    mock_asora = mock_asora_domain_decomposition

    domain_decomposition_params = DomainDecompositionParameters(
        enabled=True,
        grouping_algorithm="morton",
        max_num_sources_per_group=4,
        morton_bits=10,
    )

    # The real single-rank communicator runs the chemistry slab collectives, so the slab
    # split must match it: one rank owning the whole grid.
    xh_int, phi_ion = call_evolve3D(
        use_gpu=True,
        rank=0,
        domain_decomposition_params=domain_decomposition_params,
        nprocs=1,
    )

    N = 32
    assert xh_int.shape == (N, N, N)
    assert phi_ion.shape == (N, N, N)

    # The domain decomposition path must drive the GPU through ASORA, never the CPU
    # C2Ray raytracer.
    mock_asora.source_data_to_device.assert_called()
    mock_asora.density_to_device.assert_called()
    mock_asora.do_all_sources.assert_called()
    mock_asora.chemistry_global_pass.assert_called()
    # Chemistry runs on the slab buffers: it must not resize the shared ASORA grid
    # buffers to the full grid (the per-group raytracing setup uses the patched
    # module-level prepare_grid_buffers instead).
    mock_asora.prepare_grid_buffers.assert_not_called()

    mock_c2ray.chemistry.global_pass.assert_not_called()
    mock_c2ray.raytracing.do_all_sources.assert_not_called()


def test_evolve3D_yes_gpu_domain_decomposition_non_root_rank(
    mock_asora_domain_decomposition,
):
    mock_asora = mock_asora_domain_decomposition

    # Record the average fraction each raytracing call receives, copied because the
    # buffers are reused across groups and iterations.
    raytraced_local_xh_av: list[np.ndarray] = []
    mock_asora.do_all_sources.side_effect = lambda *args: raytraced_local_xh_av.append(
        np.array(args[3])
    )

    with patch("pyc2ray.evolve.MPI") as mock_mpi:
        mock_mpi.COMM_WORLD.Allgatherv.side_effect = lambda sendbuf, recvbuf: recvbuf[
            0
        ].fill(0.3)
        mock_mpi.COMM_WORLD.bcast.side_effect = [False, True]
        call_evolve3D(
            use_gpu=True,
            rank=1,
            domain_decomposition_params=DomainDecompositionParameters(
                enabled=True, grouping_algorithm="morton", max_num_sources_per_group=4
            ),
        )

    # Rank 0's convergence decision ended the loop after two iterations.
    assert mock_mpi.COMM_WORLD.bcast.call_count == 2

    # Every rank solves chemistry on its own slab: with 8 ranks on the 32^3 grid, rank 1
    # owns planes 4 to 7, i.e. 4 * 32 * 32 cells, in both iterations.
    assert mock_asora.chemistry_global_pass.call_count == 2
    for call in mock_asora.chemistry_global_pass.call_args_list:
        for array in call.args[1:7]:
            assert array.size == 4 * 32 * 32

    # The average fraction is gathered once, after the first (non-converged) iteration,
    # and the final fraction once after the loop.
    assert mock_mpi.COMM_WORLD.Allgatherv.call_count == 2

    # Ranks only fill their own slab of the average fraction, so it must never be read
    # before the gather completes it: the first iteration raytraces the initial
    # fraction, the second the gathered one.
    num_groups = len(raytraced_local_xh_av) // 2
    assert num_groups > 0 and len(raytraced_local_xh_av) == 2 * num_groups
    for local_xh_av in raytraced_local_xh_av[:num_groups]:
        # The initial ionized fraction set in call_evolve3D.
        np.testing.assert_array_equal(local_xh_av, 1.2e-3)
    for local_xh_av in raytraced_local_xh_av[num_groups:]:
        # The value gathered by the mocked Allgatherv.
        np.testing.assert_array_equal(local_xh_av, 0.3)


def test_evolve3D_yes_gpu_domain_decomposition_root_rank_global_convergence(
    mock_asora_domain_decomposition,
):
    mock_asora = mock_asora_domain_decomposition

    # Rank 0 of 8 owns 4 * 32 * 32 cells. Its chemistry returns a non-converged count and
    # fills its xh_int slab, while the Reduce stub adds the other ranks' totals. The values
    # are chosen so that local-only and global totals give opposite decisions on both
    # criteria (conv_criterion is 3 here):
    # - iteration 1: local count 1 would converge, the global 6 must not;
    # - iteration 2: the local sum changes by a third, but the global sum is unchanged, so
    #   only the global totals converge on the relative change.
    local_counts, local_fractions = iter([1, 4]), iter([0.5, 0.75])
    other_ranks_counts, other_ranks_sums = iter([5, 5]), iter([10000.0, 8976.0])

    def chemistry(*args):
        args[4].fill(next(local_fractions))  # xh_int slab
        return next(local_counts)

    def reduce_from_other_ranks(sendbuf, recvbuf, op, root):
        recvbuf += (next(other_ranks_counts), next(other_ranks_sums))

    # Record rank 0's decisions. The loop is ended after the two scripted iterations
    # whatever rank 0 decided, so a wrong decision fails the assertion below instead of
    # running out of scripted values.
    decisions: list[bool] = []

    def broadcast_decision(converged, root=0):
        decisions.append(converged)
        return converged or len(decisions) == 2

    mock_asora.chemistry_global_pass.side_effect = chemistry

    with patch("pyc2ray.evolve.MPI") as mock_mpi:
        mock_mpi.COMM_WORLD.Reduce.side_effect = reduce_from_other_ranks
        mock_mpi.COMM_WORLD.bcast.side_effect = broadcast_decision
        call_evolve3D(
            use_gpu=True,
            rank=0,
            domain_decomposition_params=DomainDecompositionParameters(
                enabled=True, grouping_algorithm="morton", max_num_sources_per_group=4
            ),
        )

    assert decisions == [False, True]


def test_evolve3D_yes_gpu_domain_decomposition_uneven_slabs(
    mock_asora_domain_decomposition,
):
    mock_asora = mock_asora_domain_decomposition

    # 32 planes over 6 ranks do not divide evenly: ranks 0-3 get 5 planes, ranks 4-5
    # get 6. This test runs as rank 4, which owns planes 20 to 25.
    N = 32
    planes_per_rank = [5, 5, 5, 5, 6, 6]
    first_plane_of_rank = [0, 5, 10, 15, 20, 26]
    my_planes = slice(20, 26)

    # Input fields: every cell gets a different value, so the code sees different numbers
    # whenever it looks at the wrong cells. The fields are stored in Fortran order, as in the
    # simulation.
    cell_number = np.arange(N**3, dtype=np.float64).reshape(N, N, N)
    xh = np.asfortranarray(1e-3 + 1e-9 * cell_number)
    ndens = np.asfortranarray(1e-4 + 1e-10 * cell_number)
    temp = np.asfortranarray(1e4 + cell_number)
    clump = np.asfortranarray(1.0 + 1e-6 * cell_number)
    phi_ion = 1e-12 * (1.0 + cell_number)

    # Fake raytracing sum
    def fake_allreduce(sendbuf, recvbuf, op):
        summed_phi_ion = recvbuf[0]
        summed_phi_ion[:] = phi_ion

    # Fake chemistry: records the arrays it receives, then writes results that
    # depend on its own cells. Average fraction = 2 * xh and final fraction = 3 * xh.
    chemistry_received = []

    def fake_chemistry(dt, temp, xh, xh_av, xh_int, phi_ion, clump, *constants):
        chemistry_received.append(
            {
                "temp": temp.copy(),
                "xh": xh.copy(),
                "xh_av": xh_av.copy(),
                "phi_ion": phi_ion.copy(),
                "clump": clump.copy(),
            }
        )
        xh_av[:] = 2 * xh
        xh_int[:] = 3 * xh
        return 1

    # Fake gather: the other five ranks, copies their planes of the result into the buffer, at
    # the planes the code asks for, and leaves (this) rank 4's planes alone (MPI_IN_PLACE).
    # The first gather is the average fraction, the second the final fraction.
    gathers = []

    def fake_allgatherv(sendbuf, recvbuf):
        buffer, (counts, first_planes), datatype = recvbuf
        gathers.append(
            {"counts": counts, "first_planes": first_planes, "type": datatype}
        )
        result = 2 * xh if len(gathers) == 1 else 3 * xh
        buffer_planes = buffer.reshape(N, N, N)  # same memory as buffer
        for other_rank in [0, 1, 2, 3, 5]:
            planes = slice(
                first_planes[other_rank], first_planes[other_rank] + counts[other_rank]
            )
            buffer_planes[planes] = result[planes]

    # Fake raytracing kernel: records the average fraction each local grid
    # is raytraced with.
    raytracing_received = []

    def fake_raytracing(R_max_LLS, sigma, dr, local_xh_av, *other_arguments):
        raytracing_received.append(local_xh_av.copy())

    mock_asora.chemistry_global_pass.side_effect = fake_chemistry
    mock_asora.do_all_sources.side_effect = fake_raytracing

    with patch("pyc2ray.evolve.MPI") as mock_mpi:
        mock_mpi.COMM_WORLD.Allreduce.side_effect = fake_allreduce
        mock_mpi.COMM_WORLD.Allgatherv.side_effect = fake_allgatherv
        # Not converged after the first iteration, converged after the second.
        mock_mpi.COMM_WORLD.bcast.side_effect = [False, True]
        final_xh, _ = call_evolve3D(
            use_gpu=True,
            rank=4,
            domain_decomposition_params=DomainDecompositionParameters(
                enabled=True, grouping_algorithm="morton", max_num_sources_per_group=4
            ),
            nprocs=6,
            fields={"xh": xh, "ndens": ndens, "temp": temp, "clump": clump},
        )

    # Check 1: what has been gathered is correct
    mock_mpi.DOUBLE.Create_contiguous.assert_called_once_with(N * N)
    plane = mock_mpi.DOUBLE.Create_contiguous.return_value.Commit.return_value
    assert len(gathers) == 2
    for gather in gathers:
        assert gather["counts"] == planes_per_rank
        assert gather["first_planes"] == first_plane_of_rank
        assert gather["type"] == plane

    # Check 2: chemistry works on exactly planes 20 to 25.
    # The average fraction is the initial one in the first iteration and the gathered
    # one (2 * xh) in the second.
    assert len(chemistry_received) == 2
    for iteration, received in enumerate(chemistry_received):
        np.testing.assert_array_equal(received["temp"], np.ravel(temp[my_planes]))
        np.testing.assert_array_equal(received["xh"], np.ravel(xh[my_planes]))
        expected_xh_av = xh if iteration == 0 else 2 * xh
        np.testing.assert_array_equal(
            received["xh_av"], np.ravel(expected_xh_av[my_planes])
        )
        np.testing.assert_array_equal(received["phi_ion"], np.ravel(phi_ion[my_planes]))
        np.testing.assert_array_equal(received["clump"], np.ravel(clump[my_planes]))
    # The density of planes 20 to 25 is copied to the GPU for chemistry.
    densities_sent = [
        call.args[0] for call in mock_asora.density_to_device.call_args_list
    ]
    assert any(np.array_equal(d, np.ravel(ndens[my_planes])) for d in densities_sent)

    # Check 3: the gathered fields are complete and in the right place.
    # Each local grid is raytraced once per iteration: first with xh, then with the
    # gathered average fraction, which must be 2 * xh everywhere.
    half = len(raytracing_received) // 2
    assert half > 0 and len(raytracing_received) == 2 * half
    for first_iteration, second_iteration in zip(
        raytracing_received[:half], raytracing_received[half:]
    ):
        np.testing.assert_array_equal(second_iteration, 2 * first_iteration)
    # The returned final fraction must be 3 * xh on all 32 planes.
    np.testing.assert_array_equal(final_xh, 3 * xh)
