from contextlib import contextmanager
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from pyc2ray.load_extensions import libasora_He as libasora
from pyc2ray.radiation.blackbody import BlackBodySource_Multifreq
from pyc2ray.radiation.common import make_tau_table
from pyc2ray.radiation.radiation_tables import RadiationTables

if libasora is None:
    pytest.skip("libasora_He.so missing, skipping tests", allow_module_level=True)


@contextmanager
def setup_do_all_sources(
    data_dir: Path,
    num_sources: int = 10,
    mesh_size: int = 50,
    batch_size: int = 1,
    block_size: int = 256,
    radius: float = 15.0,
):
    # Calculate the table
    minlog_tau, maxlog_tau, num_tau = -20.0, 4.0, 2000
    tau, dlogtau = make_tau_table(minlog_tau, maxlog_tau, num_tau)

    # Calculate the table
    radsource = BlackBodySource_Multifreq(1e5)
    photo_thin_table, photo_thick_table = radsource.make_photo_tables(tau)
    heat_thin_table, heat_thick_table = radsource.make_heat_tables(tau)

    # Read cross section
    rt = RadiationTables()
    sigmas = rt.cross_sections
    heat_factors = rt.factors
    nfreq = len(sigmas[0])

    assert photo_thin_table.shape == (nfreq, num_tau + 1)

    # Allocate tables to GPU device
    assert libasora is not None
    libasora.photo_tables_to_device(
        photo_thin_table.ravel(),
        photo_thick_table.ravel(),
        heat_thin_table.ravel(),
        heat_thick_table.ravel(),
    )

    size = mesh_size**3

    phion_HI = np.empty(size, dtype=np.float64)
    phion_HeI = np.empty(size, dtype=np.float64)
    phion_HeII = np.empty(size, dtype=np.float64)
    pheat = np.empty(size, dtype=np.float64)

    ndens = np.full(size, 1.0e-6, dtype=np.float64)
    xHII = np.full_like(ndens, 1.0e-3)
    xHeII = np.full_like(ndens, 1.0e-3)
    xHeIII = np.full_like(ndens, 1.0e-3)

    # Copy density field to GPU device
    assert libasora is not None
    libasora.density_to_device(ndens)

    # Efficiency factor (converting mass to photons)
    f_gamma = 100.0

    # Define some random sources
    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, mesh_size, size=(3 * num_sources), dtype=np.int32)
    norm_flux = rng.uniform(1e10, 1e14, size=num_sources).astype(np.float64)
    norm_flux *= f_gamma / 1e48

    # Copy source list to GPU device
    assert libasora is not None
    libasora.source_data_to_device(src_pos, norm_flux)

    # Size of a cell
    boxsize = 1.5 * u.Mpc
    dr = (boxsize / mesh_size).cgs.value

    yield (
        radius,
        *sigmas,
        heat_factors,
        nfreq,
        dr,
        xHII,
        xHeII,
        xHeIII,
        phion_HI,
        phion_HeI,
        phion_HeII,
        pheat,
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
        libasora.do_all_sources(*args)

        phion_HI = args[10] * 1e48
        phion_HeI = args[11] * 1e48
        phion_HeII = args[12] * 1e48
        pheat = args[13] * 1e48

        expected_rates = np.load(data_dir / "photo_rates_with_helium.npz")

        assert np.allclose(phion_HI, expected_rates["ion_HI"])
        assert np.allclose(phion_HeI, expected_rates["ion_HeI"])
        assert np.allclose(phion_HeII, expected_rates["ion_HeII"])
        assert np.allclose(pheat, expected_rates["heat"])
