from contextlib import contextmanager

import numpy as np
from astropy import constants as cst
from astropy import units as u

import pyc2ray.solver as pysolver
from pyc2ray.load_extensions import libc2ray


def test_load_c2ray():
    assert libc2ray is not None


@contextmanager
def setup_chemistry(mesh_size: int = 10):
    mesh_shape = (mesh_size,) * 3
    rng = np.random.default_rng(2023)

    # time-step
    dt = (50 * u.yr).to("s").value

    # density field [g/cm^3]
    ndens = rng.normal(1e-7, 1e-8, size=mesh_shape).astype(np.float64, order="F")

    # temperature [K]
    temp = np.full(mesh_shape, 1e4, dtype=np.float64, order="F")

    # Hydrogen ionization fraction
    xh = rng.uniform(0, 0.1, size=mesh_shape).astype(np.float64, order="F")
    xh_av = xh.copy(order="F")
    xh_int = xh.copy(order="F")

    # photo-ionization rate [s^-1]
    phi_ion = rng.uniform(1e-13, 1e-12, size=mesh_shape).astype(np.float64, order="F")

    # clumping factor
    clump = np.ones(mesh_shape, dtype=np.float64, order="F")

    # constants
    eth0 = 13.598
    bh00 = 2.59e-13
    colh0 = 1.079e-8 / eth0**2
    albpow = -0.7
    temph0 = eth0 / (cst.k_B * u.K).to("eV").value
    abu_c = 7.1e-7

    yield (
        dt,
        ndens,
        temp,
        xh,
        xh_av,
        xh_int,
        phi_ion,
        clump,
        bh00,
        albpow,
        colh0,
        temph0,
        abu_c,
    )


def test_chemistry(data_dir):
    with setup_chemistry() as args:
        xh = args[3]
        xh_int = args[5]

        for _ in range(1000):
            libc2ray.chemistry.global_pass(*args)
            xh[:] = xh_int

        expected_xh = np.load(data_dir / "ionized_fraction_average.npy")
        assert np.allclose(xh, expected_xh)


def test_chemistry_python(data_dir):
    with setup_chemistry() as args:
        xh = args[3]

        for _ in range(1000):
            new_xh, _, _ = pysolver.chemistry.global_pass(*args)
            xh[:] = new_xh

        expected_xh = np.load(data_dir / "ionized_fraction_average.npy")
        assert np.allclose(xh, expected_xh)


def test_benchmark_chemistry(benchmark, data_dir):
    with setup_chemistry() as args:
        benchmark(libc2ray.chemistry.global_pass, *args)


def test_benchmark_chemistry_python(benchmark, data_dir):
    with setup_chemistry() as args:
        benchmark(pysolver.chemistry.global_pass, *args)
