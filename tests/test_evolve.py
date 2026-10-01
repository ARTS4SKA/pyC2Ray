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
        nprocs=8,
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
    mock_asora.prepare_grid_buffers.assert_called()

    mock_c2ray.chemistry.global_pass.assert_not_called()
    mock_c2ray.raytracing.do_all_sources.assert_not_called()
