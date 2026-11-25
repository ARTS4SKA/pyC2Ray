from collections.abc import Generator
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
def mock_c2ray() -> Generator[Mock]:
    mock = Mock()
    mock.chemistry.global_pass.return_value = 1
    mock.raytracing.do_all_sources.return_value = (10, 0.1)
    with patch("pyc2ray.evolve.libc2ray", mock):
        yield mock


@pytest.fixture
def mock_asora() -> Generator[Mock]:
    # check_device_init decorates the evolve entry points at import time, so patching
    # the decorator is too late. It looks libasora up in the asora_core globals on
    # every call, so replacing that is what disarms the guard. The same mock goes into
    # the evolve namespace as well, since some calls reach the extension module
    # directly and others go through the asora_core wrappers; one mock means one place
    # to assert on, and the wrappers themselves stay unmocked.
    mock = Mock()
    mock.is_device_init.return_value = True
    mock.chemistry_global_pass.return_value = 1
    mock.num_column_density_banks.return_value = 4
    with (
        patch("pyc2ray.evolve.libasora", mock),
        patch("pyc2ray.evolve.libasora", mock),
        patch("pyc2ray.asora_core.libasora", mock),
    ):
        yield mock


def call_evolve3D(
    use_gpu: bool = False,
    use_mpi: bool = False,
    rank: int = 0,
    domain_decomposition_params: DomainDecompositionParameters | None = None,
    fields: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    N = 32

    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, N, size=(3, 10), dtype=np.int32)
    src_flux = rng.uniform(1e10, 1e14, size=10).astype(np.float64)
    src_flux *= 1e-46

    # The domain decomposition path builds a per-group local grid sized by the
    # sources' radius of influence (R_max cells). On this small test grid a
    # radius of 15 cells would make local grids larger than the global grid,
    # which the periodic mapping forbids. Use a smaller radius in that case.
    if domain_decomposition_params is not None:
        R_max = 2.0
    else:
        R_max = 15.0

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
            R_max=R_max,
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
        use_mpi=use_mpi,
        rank=rank,
        nprocs=8,
        temp=temp,
        ndens=ndens,
        xh=xh,
        clump=clump,
        photo_thin_table=photo_thin_table,
        photo_thick_table=photo_thick_table,
        minlogtau=minlogtau,
        dlogtau=dlogtau,
        R_max=R_max,
        convergence_fraction=1e-4,
        sigma=sigma,
        chems=ChemistryParams(2.59e-13, -0.7, colh0, temph0, 7.1e-7),
        decomposition=decomposition,
    )


def test_evolve3D_no_gpu_root_rank(mock_c2ray: Mock, mock_asora: Mock) -> None:
    call_evolve3D(use_gpu=False, rank=0)

    mock_asora.source_data_to_device.assert_not_called()
    mock_asora.density_to_device.assert_not_called()
    mock_asora.do_all_sources.assert_not_called()
    mock_asora.chemistry_global_pass.assert_not_called()
    mock_asora.device_trim.assert_not_called()

    mock_c2ray.chemistry.global_pass.assert_called()
    mock_c2ray.raytracing.do_all_sources.assert_called()


def test_evolve3D_yes_gpu_root_rank(mock_c2ray: Mock, mock_asora: Mock) -> None:
    call_evolve3D(use_gpu=True, rank=0)

    mock_asora.source_data_to_device.assert_called()
    mock_asora.density_to_device.assert_called()
    mock_asora.do_all_sources.assert_called()
    mock_asora.chemistry_global_pass.assert_called()
    mock_asora.device_trim.assert_called()

    mock_c2ray.chemistry.global_pass.assert_not_called()
    mock_c2ray.raytracing.do_all_sources.assert_not_called()


def test_evolve3D_yes_gpu_domain_decomposition_root_rank(
    mock_c2ray: Mock, mock_asora: Mock
) -> None:

    domain_decomposition_params = DomainDecompositionParameters.from_dict(
        {
            "enabled": True,
            "grouping_algorithm": "morton",
            "max_num_sources_per_group": 4,
            "morton_bits": 10,
        }
    )

    xh_int, phi_ion = call_evolve3D(
        use_gpu=True,
        rank=0,
        domain_decomposition_params=domain_decomposition_params,
    )

    N = 32
    assert xh_int.shape == (N, N, N)
    assert phi_ion.shape == (N, N, N)

    # The domain decomposition path must drive the GPU through ASORA and use
    # the per-group grid buffer setup, never the CPU C2Ray raytracer.
    mock_asora.source_data_to_device.assert_called()
    mock_asora.density_to_device.assert_called()
    mock_asora.do_all_sources.assert_called()
    mock_asora.chemistry_global_pass.assert_called()
    mock_asora.device_trim.assert_called()

    mock_c2ray.chemistry.global_pass.assert_not_called()
    mock_c2ray.raytracing.do_all_sources.assert_not_called()


def test_evolve3D_yes_gpu_domain_decomposition_non_root_rank(mock_asora: Mock) -> None:
    # Record the average fraction sent to the device, copied because the buffers are
    # reused across groups and iterations. Raytracing sends each 3D local grid,
    # chemistry the flat slab of this rank.
    sent_xh_av: list[np.ndarray] = []
    with patch("pyc2ray.evolve.average_fraction_to_device") as mock_xh_av:
        mock_xh_av.side_effect = lambda xh_av: sent_xh_av.append(np.array(xh_av))

        with patch("pyc2ray.evolve.comm") as mock_mpi:
            mock_mpi.Allreduce.return_value = None
            mock_mpi.Allgatherv.side_effect = lambda sendbuf, recvbuf: recvbuf[0].fill(
                0.3
            )
            mock_mpi.Get_size.return_value = 8
            mock_mpi.bcast.side_effect = [False, True]
            call_evolve3D(
                use_gpu=True,
                use_mpi=True,
                rank=1,
                domain_decomposition_params=DomainDecompositionParameters.from_dict(
                    {
                        "enabled": True,
                        "grouping_algorithm": "morton",
                        "max_num_sources_per_group": 4,
                    }
                ),
            )

    # Rank 0's convergence decision ended the loop after two iterations.
    assert mock_mpi.bcast.call_count == 2

    # Every rank solves chemistry on its own slab: with 8 ranks on the 32^3 grid, rank 1
    # owns planes 4 to 7, i.e. 4 * 32 * 32 cells, in both iterations.
    slab_cells = 4 * 32 * 32
    assert mock_asora.chemistry_global_pass.call_count == 2
    for call in mock_asora.chemistry_global_pass.call_args_list:
        xh_int, phi_ion = call.args[1], call.args[2]
        assert xh_int.size == slab_cells and phi_ion.size == slab_cells
    chemistry_xh_av = [xh_av for xh_av in sent_xh_av if xh_av.ndim == 1]
    assert [xh_av.size for xh_av in chemistry_xh_av] == [slab_cells, slab_cells]

    # The average fraction is gathered once, after the first (non-converged) iteration,
    # and the final fraction once after the loop.
    assert mock_mpi.Allgatherv.call_count == 2

    # The first iteration raytraces the initial fraction, the second the gathered one.
    raytraced_xh_av = [xh_av for xh_av in sent_xh_av if xh_av.ndim == 3]
    num_groups = len(raytraced_xh_av) // 2
    assert num_groups > 0 and len(raytraced_xh_av) == 2 * num_groups

    # The initial ionized fraction set in call_evolve3D.
    for local_xh_av in raytraced_xh_av[:num_groups]:
        np.testing.assert_array_equal(local_xh_av, 1.2e-3)

    # The value gathered by the mocked Allgatherv.
    for local_xh_av in raytraced_xh_av[num_groups:]:
        np.testing.assert_array_equal(local_xh_av, 0.3)


def test_evolve3D_yes_gpu_domain_decomposition_root_rank_global_convergence(
    mock_asora: Mock,
) -> None:
    # Rank 0 of 8 owns 4 * 32 * 32 cells. Its chemistry returns a non-converged count and
    # fills its xh_int slab, while the Reduce stub adds the other ranks' totals. The values
    # are chosen so that local-only and global totals give opposite decisions on both
    # criteria (conv_criterion is 3 here):
    # - iteration 1: local count 1 would converge, the global 6 must not;
    # - iteration 2: the local sum changes by a third, but the global sum is unchanged, so
    #   only the global totals converge on the relative change.
    local_counts, local_fractions = iter([1, 4]), iter([0.5, 0.75])
    other_ranks_counts, other_ranks_sums = iter([5, 5]), iter([10000.0, 8976.0])

    def fake_chemistry(dt, xh_int, phi_ion, *constants):
        xh_int.fill(next(local_fractions))
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

    mock_asora.chemistry_global_pass.side_effect = fake_chemistry

    with patch("pyc2ray.evolve.comm") as mock_mpi:
        mock_mpi.Reduce.side_effect = reduce_from_other_ranks
        mock_mpi.Get_size.return_value = 8
        mock_mpi.bcast.side_effect = broadcast_decision
        call_evolve3D(
            use_gpu=True,
            use_mpi=True,
            rank=0,
            domain_decomposition_params=DomainDecompositionParameters.from_dict(
                {
                    "enabled": True,
                    "grouping_algorithm": "morton",
                    "max_num_sources_per_group": 4,
                }
            ),
        )

    assert decisions == [False, True]


def test_evolve3D_yes_gpu_domain_decomposition_uneven_slabs(mock_asora: Mock) -> None:
    # 32 planes over 6 ranks do not divide evenly: ranks 0-3 get 5 planes, ranks 4-5
    # get 6. This test runs as rank 4, which owns planes 20 to 25.
    N = 32
    planes_per_rank = [5, 5, 5, 5, 6, 6]
    first_plane_of_rank = [0, 5, 10, 15, 20, 26]
    my_planes = slice(20, 26)

    # Every cell gets a different value, so the code sees different numbers whenever it
    # looks at the wrong cells. The fields are stored in Fortran order, as in the
    # simulation.
    cell_number = np.arange(N**3, dtype=np.float64).reshape(N, N, N)
    xh = np.asfortranarray(1e-3 + 1e-9 * cell_number)
    ndens = np.asfortranarray(1e-4 + 1e-10 * cell_number)
    temp = np.asfortranarray(1e4 + cell_number)
    clump = np.asfortranarray(1.0 + 1e-6 * cell_number)
    phi_ion = 1e-12 * (1.0 + cell_number)
    my_xh = np.ravel(xh[my_planes])

    # Raytracing is mocked, so the Allreduce writes the photoionization rates directly.
    def fake_allreduce(sendbuf, recvbuf, op):
        summed_phi_ion = recvbuf[0]
        summed_phi_ion[:] = phi_ion

    # Record what is sent. Raytracing sends each local grid's density (flat) and average
    # fraction (3D); chemistry sends its flat slabs.
    sent_timestep_data = []
    sent_densities = []
    sent_xh_av = []

    # Copies flattened in C order, as the real wrapper sends them.
    def fake_timestep_data_to_device(xh, temp, clump):
        sent_timestep_data.append(
            {
                "xh": np.ravel(xh).copy(),
                "temp": np.ravel(temp).copy(),
                "clump": np.ravel(clump).copy(),
            }
        )

    def fake_density_to_device(density):
        sent_densities.append(np.array(density))

    def fake_average_fraction_to_device(xh_av):
        sent_xh_av.append(np.array(xh_av))

    # Records the rates chemistry receives and writes results that depend on rank 4's cells:
    # final fraction = 3 * xh, and average fraction = 2 * xh, which stays on the device
    # until average_fraction_to_host brings it back.
    chemistry_received_phi_ion = []

    def fake_chemistry(dt, xh_int, phi_ion, *constants):
        chemistry_received_phi_ion.append(np.array(phi_ion))
        xh_int[:] = 3 * my_xh
        return 1

    def fake_average_fraction_to_host(xh_av):
        xh_av[:] = 2 * my_xh

    # Fake gather: plays the other five ranks: copies their planes of the result into the buffer, at
    # the planes the code asks for, and leaves rank 4's planes alone (MPI_IN_PLACE).
    # The first gather is the average fraction, the second the final fraction.
    gathers = []

    def fake_allgatherv(sendbuf, recvbuf):
        buffer, (counts, first_planes), datatype = recvbuf
        gathers.append(
            {
                "counts": counts,
                "first_planes": first_planes,
                "plane_bytes": datatype.Get_size(),
            }
        )
        result = 2 * xh if len(gathers) == 1 else 3 * xh
        buffer_planes = buffer.reshape(N, N, N)  # same memory as buffer
        for other_rank in [0, 1, 2, 3, 5]:
            planes = slice(
                first_planes[other_rank], first_planes[other_rank] + counts[other_rank]
            )
            buffer_planes[planes] = result[planes]

    mock_asora.chemistry_global_pass.side_effect = fake_chemistry

    with (
        patch("pyc2ray.evolve.timestep_data_to_device", fake_timestep_data_to_device),
        patch("pyc2ray.evolve.density_to_device", fake_density_to_device),
        patch(
            "pyc2ray.evolve.average_fraction_to_device", fake_average_fraction_to_device
        ),
        patch("pyc2ray.evolve.average_fraction_to_host", fake_average_fraction_to_host),
        patch("pyc2ray.evolve.comm") as mock_mpi,
    ):
        mock_mpi.Allreduce.side_effect = fake_allreduce
        mock_mpi.Allgatherv.side_effect = fake_allgatherv
        mock_mpi.Get_size.return_value = 6
        # Not converged after the first iteration, converged after the second.
        mock_mpi.bcast.side_effect = [False, True]
        final_xh, _ = call_evolve3D(
            use_gpu=True,
            use_mpi=True,
            rank=4,
            domain_decomposition_params=DomainDecompositionParameters.from_dict(
                {
                    "enabled": True,
                    "grouping_algorithm": "morton",
                    "max_num_sources_per_group": 4,
                }
            ),
            fields={"xh": xh, "ndens": ndens, "temp": temp, "clump": clump},
        )

    # Check 1: the gathers use every rank's planes
    # One MPI element is one plane of N * N doubles.
    assert len(gathers) == 2
    for gather in gathers:
        assert gather["counts"] == planes_per_rank
        assert gather["first_planes"] == first_plane_of_rank
        assert gather["plane_bytes"] == N * N * 8

    # Check 2: chemistry works on exactly planes 20 to 25
    # The fields that do not change over the timestep are sent once.
    assert len(sent_timestep_data) == 1
    np.testing.assert_array_equal(sent_timestep_data[0]["xh"], my_xh)
    np.testing.assert_array_equal(
        sent_timestep_data[0]["temp"], np.ravel(temp[my_planes])
    )
    np.testing.assert_array_equal(
        sent_timestep_data[0]["clump"], np.ravel(clump[my_planes])
    )
    # In both iterations: the slab density, the slab average fraction (the initial one,
    # then the gathered one, 2 * xh) and the slab rates.
    assert any(np.array_equal(d, np.ravel(ndens[my_planes])) for d in sent_densities)
    chemistry_xh_av = [xh_av for xh_av in sent_xh_av if xh_av.ndim == 1]
    assert len(chemistry_xh_av) == 2
    np.testing.assert_array_equal(chemistry_xh_av[0], my_xh)
    np.testing.assert_array_equal(chemistry_xh_av[1], 2 * my_xh)
    assert len(chemistry_received_phi_ion) == 2
    for received in chemistry_received_phi_ion:
        np.testing.assert_array_equal(received, np.ravel(phi_ion[my_planes]))

    # Check 3: the gathered fields are complete and in the right place
    # Each local grid is raytraced once per iteration: first with xh, then with the
    # gathered average fraction, which must be 2 * xh everywhere.
    raytraced_xh_av = [xh_av for xh_av in sent_xh_av if xh_av.ndim == 3]
    half = len(raytraced_xh_av) // 2
    assert half > 0 and len(raytraced_xh_av) == 2 * half
    for first_iteration, second_iteration in zip(
        raytraced_xh_av[:half], raytraced_xh_av[half:]
    ):
        np.testing.assert_array_equal(second_iteration, 2 * first_iteration)
    # The returned final fraction must be 3 * xh on all 32 planes.
    np.testing.assert_array_equal(final_xh, 3 * xh)
