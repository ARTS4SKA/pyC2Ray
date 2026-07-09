from contextlib import contextmanager
from unittest.mock import Mock, patch

import astropy.constants as cst
import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import Planck18 as cosmo
from mpi4py import MPI

from pyc2ray.evolve import ChemistryParams, evolve3D
from pyc2ray.load_extensions import libasora_He as libasora
from pyc2ray.radiation.blackbody import BlackBodySource_Multifreq
from pyc2ray.radiation.common import make_tau_table
from pyc2ray.solver.helium import CoolingTables


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
        mock.configure_mock(
            **{
                "chemistry_global_pass": Mock(return_value=1),
            }
        )
        yield mock


@pytest.fixture(scope="module")
def chem_params() -> ChemistryParams:
    colh0 = 1.3e-8 * 0.83 * 1.0 / 13.598**2
    temph0 = 13.598 / (cst.k_B * u.K).to("eV").value
    return ChemistryParams(2.59e-13, -0.7, colh0, temph0, 7.1e-7)


@contextmanager
def setup_evolve_mock(use_gpu: bool = False, rank: int = 0):
    Hz = cosmo.H(10.0).cgs.value
    N = 32

    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, N, size=(3, 10), dtype=np.int32)
    src_flux = rng.uniform(1e10, 1e14, size=10).astype(np.float64)
    src_flux *= 1e-46

    shape = (N, N, N)
    ndens = np.empty(shape, dtype=np.float64, order="F")
    xh = (
        np.full_like(ndens, 2.0e-4),
        np.full_like(ndens, 1.0e-15),
        np.full_like(ndens, 0.0),
    )
    temp = np.full_like(ndens, 1e4)
    clump = np.full_like(ndens, 1.0)

    minlogtau, maxlogtau, num_tau = -20.0, 4.0, 20000
    tau, dlogtau = make_tau_table(minlogtau, maxlogtau, num_tau)
    logtau = minlogtau, dlogtau, num_tau
    sigma = 6.3e-18
    photo_thin_table = np.zeros(num_tau, dtype=np.float64)
    photo_thick_table = np.zeros(num_tau, dtype=np.float64)

    yield dict(
        Hz=Hz,
        dt=1e3,
        dr=(1 * u.Mpc).cgs.value / N,
        R_max=15.0,
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
        clump=clump,
        xh=xh,
        photo_thin_table=photo_thin_table,
        photo_thick_table=photo_thick_table,
        convergence_fraction=1e-4,
        sigma=sigma,
        logtau=logtau,
        logtemp=(1.0, 0.1, 100),
    )


def test_evolve3D_no_gpu_root_rank(mock_c2ray, mock_asora, chem_params):
    with setup_evolve_mock(use_gpu=False, rank=0) as kwargs:
        evolve3D(**kwargs, chems=chem_params)

    mock_asora.source_data_to_device.assert_not_called()
    mock_asora.density_to_device.assert_not_called()
    mock_asora.do_all_sources.assert_not_called()
    mock_asora.chemistry_global_pass.assert_not_called()

    mock_c2ray.chemistry.global_pass.assert_called()
    mock_c2ray.raytracing.do_all_sources.assert_called()


def test_evolve3D_yes_gpu_root_rank(mock_c2ray, mock_asora, chem_params):
    with setup_evolve_mock(use_gpu=True, rank=0) as kwargs:
        evolve3D(**kwargs, chems=chem_params)

    mock_asora.source_data_to_device.assert_called()
    mock_asora.density_to_device.assert_called()
    mock_asora.do_all_sources.assert_called()
    mock_asora.chemistry_global_pass.assert_called()

    mock_c2ray.chemistry.global_pass.assert_not_called()
    mock_c2ray.raytracing.do_all_sources.assert_not_called()


@contextmanager
def setup_evolve_asora(
    mesh_size: int = 50,
    num_sources: int = 10,
    radius: float = 15.0,
):
    # Calculate the table
    minlog_tau, maxlog_tau, num_tau = -20.0, 4.0, 20000
    tau, dlogtau = make_tau_table(minlog_tau, maxlog_tau, num_tau)
    logtau = minlog_tau, dlogtau, num_tau

    # Calculate the table
    radsource = BlackBodySource_Multifreq(1e5)
    photo_thin_table, photo_thick_table = radsource.make_photo_tables(tau)
    heat_thin_table, heat_thick_table = radsource.make_heat_tables(tau)

    cool_tables = CoolingTables.from_dir()

    # Allocate tables to GPU device
    assert libasora is not None

    libasora.photo_tables_to_device(
        photo_thin_table.ravel(),
        photo_thick_table.ravel(),
        heat_thin_table.ravel(),
        heat_thick_table.ravel(),
    )

    libasora.cooling_tables_to_device(
        cool_tables.HI,
        cool_tables.HII,
        cool_tables.HeI,
        cool_tables.HeII,
        cool_tables.HeIII,
    )

    mesh_shape = mesh_size, mesh_size, mesh_size

    # density field [g/cm^3]
    rng = np.random.default_rng(1111)
    ndens = rng.uniform(1e-6, 1e-3, size=mesh_shape).astype(np.float64)

    # temperature [K]
    temp = np.pow(10, rng.normal(4, 0.25, size=mesh_shape), dtype=np.float64)
    clump = np.ones_like(temp)

    xHII = np.full_like(ndens, 2.0e-4)
    xHeII = np.full_like(ndens, 1.0e-15)
    xHeIII = np.full_like(ndens, 0.0)

    # Define some random sources
    src_pos = rng.integers(0, mesh_size, size=(3, num_sources), dtype=np.int32)
    src_flux = rng.uniform(1e10, 1e14, size=num_sources).astype(np.float64)
    src_flux *= 100.0 / 1e48

    # Size of a cell
    dr = (1.5 * u.Mpc / mesh_size).cgs.value
    dt = (5 * u.Myr).cgs.value
    Hz = cosmo.H(10.0).cgs.value

    comm = MPI.COMM_WORLD
    yield dict(
        Hz=Hz,
        dr=dr,
        dt=dt,
        R_max=radius,
        src_flux=src_flux,
        src_pos=src_pos,
        src_batch_size=8,
        temp=temp,
        ndens=ndens,
        clump=clump,
        xh=(xHII, xHeII, xHeIII),
        convergence_fraction=1e-4,
        use_gpu=True,
        use_mpi=True,
        rank=comm.Get_rank(),
        nprocs=comm.Get_size(),
        logtau=logtau,
        logtemp=cool_tables.logtemp,
    )


def test_evolve_asora(init_device, chem_params):
    with setup_evolve_asora() as kwargs:
        shape = kwargs["ndens"].shape
        xh, phion, pheat, temp = evolve3D(**kwargs, chems=chem_params)

    assert len(xh) == 3
    assert xh[0].shape == shape
    assert xh[1].shape == shape
    assert xh[2].shape == shape

    assert len(phion) == 3
    assert phion[0].shape == shape
    assert phion[1].shape == shape
    assert phion[2].shape == shape

    assert len(pheat) == 3
    assert pheat[0].shape == shape
    assert pheat[1].shape == shape
    assert pheat[2].shape == shape

    assert temp.shape == shape
