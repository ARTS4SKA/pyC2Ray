from contextlib import contextmanager
from pathlib import Path

import astropy.constants as cst
import astropy.units as u
import numpy as np
import pytest

from pyc2ray.load_extensions import load_asora_he
from pyc2ray.radiation.blackbody import BlackBodySource_Multifreq
from pyc2ray.radiation.common import make_tau_table

asora = load_asora_he()

if asora is None:
    pytest.skip("libasoraHe.so missing, skipping tests", allow_module_level=True)


@pytest.fixture
def init_device():
    asora.device_init()
    yield
    asora.device_close()


def test_device_init(init_device):
    asora.is_device_init()
    pass


@contextmanager
def setup_do_all_sources(
    data_dir: Path,
    num_sources: int = 10,
    mesh_size: int = 50,
    batch_size: int = 1,
    block_size: int = 256,
):
    R_max = 15.0

    # HI cross section at its ionzing frequency (weighted by freq_factor)
    # Calculate the table
    minlog_tau, maxlog_tau, num_tau = -20.0, 4.0, 20000
    tau, dlogtau = make_tau_table(minlog_tau, maxlog_tau, num_tau)

    # Min and max frequency of the integral
    freq_min, freq_max = (
        (13.598 * u.eV / cst.h).to("Hz").value,
        (54.416 * u.eV / cst.h).to("Hz").value,
    )

    # Calculate the table
    radsource = BlackBodySource_Multifreq(1e5, False)
    photo_thin_table, photo_thick_table = radsource.make_photo_table(
        tau, freq_min, freq_max, 1e48
    )
    heat_thin_table, heat_thick_table = radsource.make_heat_table(
        tau, freq_min, freq_max, 1e48
    )

    # Read cross section
    _, sigma_HI, sigma_HeI, sigma_HeII = np.loadtxt(
        data_dir / "Verner1996_crossect.txt", unpack=True
    )
    sigma_HI = sigma_HI.ravel()
    sigma_HeI = sigma_HeI.ravel()
    sigma_HeII = sigma_HeII.ravel()

    # number of frequency bin
    numb1, numb2, numb3 = 1, 26, 20
    num_freq = numb1 + numb2 + numb3

    assert photo_thin_table.shape[0] == num_freq

    # Allocate tables to GPU device
    asora.tables_to_device(
        photo_thin_table.ravel(),
        photo_thick_table.ravel(),
        heat_thin_table.ravel(),
        heat_thick_table.ravel(),
    )

    size = mesh_size**3

    phion_HI = np.zeros(size, dtype=np.float64)
    phion_HeI = np.zeros(size, dtype=np.float64)
    phion_HeII = np.zeros(size, dtype=np.float64)
    pheat_HI = np.zeros(size, dtype=np.float64)
    pheat_HeI = np.zeros(size, dtype=np.float64)
    pheat_HeII = np.zeros(size, dtype=np.float64)

    ndens = np.full(size, 1.87e-7, dtype=np.float64)
    xHI = np.full(size, 1.2e-3, dtype=np.float64)
    xHeI = np.full(size, 1e-3, dtype=np.float64)
    xHeII = np.full(size, 1e-3, dtype=np.float64)

    # Copy density field to GPU device
    asora.density_to_device(ndens)

    # Efficiency factor (converting mass to photons)
    f_gamma = 100.0

    # Define some random sources
    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, mesh_size, size=(3 * num_sources), dtype=np.int32)
    norm_flux = rng.uniform(1e10, 1e14, size=num_sources).astype(np.float64)
    norm_flux *= f_gamma / 1e48

    # Copy source list to GPU device
    print(src_pos.reshape(num_sources, 3))
    asora.source_data_to_device(src_pos, norm_flux)

    # Size of a cell
    boxsize = 1.62022035 * u.Mpc
    dr = (boxsize / mesh_size).cgs.value

    yield (
        R_max,
        sigma_HI,
        sigma_HeI,
        sigma_HeII,
        numb1,
        numb2,
        numb3,
        num_freq,
        dr,
        xHI,
        xHeI,
        xHeII,
        phion_HI,
        phion_HeI,
        phion_HeII,
        pheat_HI,
        pheat_HeI,
        pheat_HeII,
        num_sources,
        mesh_size,
        minlog_tau,
        dlogtau,
        num_tau,
        batch_size,
        block_size,
    )


def test_do_all_sources(data_dir, init_device):
    with setup_do_all_sources(data_dir) as args:
        asora.do_all_sources(*args)

        phion_HI = args[12] * 1e48
        phion_HeI = args[13] * 1e48
        phion_HeII = args[14] * 1e48
        pheat_HI = args[15] * 1e48
        pheat_HeI = args[16] * 1e48
        pheat_HeII = args[17] * 1e48

        if False:
            np.savez(
                data_dir / "photo_rates_with_helium.npz",
                ion_HI=phion_HI,
                ion_HeI=phion_HeI,
                ion_HeII=phion_HeII,
                heat_HI=pheat_HI,
                heat_HeI=pheat_HeI,
                heat_HeII=pheat_HeII,
            )

        expected_rates = np.load(data_dir / "photo_rates_with_helium.npz")

        assert np.allclose(phion_HI, expected_rates["ion_HI"])
        assert np.allclose(phion_HeI, expected_rates["ion_HeI"])
        assert np.allclose(phion_HeII, expected_rates["ion_HeII"])
        assert np.allclose(pheat_HI, expected_rates["heat_HI"], equal_nan=True)
        assert np.allclose(pheat_HeI, expected_rates["heat_HeI"], equal_nan=True)
        assert np.allclose(pheat_HeII, expected_rates["heat_HeII"], equal_nan=True)
