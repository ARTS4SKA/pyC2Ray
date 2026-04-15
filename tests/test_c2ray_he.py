from contextlib import contextmanager

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from pyc2ray.load_extensions import libc2ray
from pyc2ray.solver.helium import thermal


def test_load_c2ray() -> None:
    assert libc2ray is not None
    assert hasattr(libc2ray, "chemistry_he")
    assert hasattr(libc2ray.chemistry_he, "thermal")


def test_thermal_evolution_only_cosmic_expansion_python() -> None:
    ti = cosmo.age(40).cgs.value
    tf = cosmo.age(2).cgs.value
    times = np.linspace(ti, tf, 200, endpoint=False)
    zreds = z_at_value(cosmo.age, times * u.s).value

    T0 = cosmo.Tcmb(zreds[0]).cgs.value

    # Get analytical solution for the temperature evolution
    temps_anal = T0 * np.pow(times / times[0], -4 / 3)

    # Get numerical solution using the thermal function
    dts = np.diff(times)
    temps_num = np.full_like(temps_anal, T0)
    for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
        Hz = cosmo.H(zi).cgs.value
        # These combinations of parameters disables heating and cooling rates:
        # the temperature evolution is only due to cosmic expansion
        T0, _ = thermal(dt, T0, 1.0, 0.0, 0.0, Hz)
        temps_num[i] = T0

    # TODO: change this tests once we can update H during the evolution.
    assert (np.abs(temps_num - temps_anal) / temps_anal < 0.75).all()


def test_thermal_evolution_only_cosmic_expansion_fortran() -> None:
    ti = cosmo.age(40).cgs.value
    tf = cosmo.age(2).cgs.value
    times = np.linspace(ti, tf, 200, endpoint=False)
    zreds = z_at_value(cosmo.age, times * u.s).value

    T_start = cosmo.Tcmb(zreds[0]).cgs.value
    T0 = np.array(T_start, dtype=np.float64)
    TA = T0.copy()

    # Get analytical solution for the temperature evolution
    temps_anal = T0 * np.pow(times / times[0], -4 / 3)

    # Get numerical solution using the thermal function
    dts = np.diff(times)
    temps_num = np.full_like(temps_anal, T0)
    for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
        Hz = cosmo.H(zi).cgs.value
        # These combinations of parameters disables heating and cooling rates:
        # the temperature evolution is only due to cosmic expansion
        libc2ray.chemistry_he.thermal(dt, T0, TA, 1.0, 0.0, 0.0, Hz)
        temps_num[i] = T0

    # TODO: change this tests once we can update H during the evolution.
    assert (np.abs(temps_num - temps_anal) / temps_anal < 0.75).all()


@contextmanager
def setup_chemistry(mesh_size: int = 10):
    mesh_shape = (mesh_size,) * 3
    rng = np.random.default_rng(2023)

    # time-step
    dt = (50 * u.yr).cgs.value
    dr = (1 * u.Mpc).cgs.value
    Hz = cosmo.H(10).cgs.value

    # density field [g/cm^3]
    ndens = rng.normal(1e-7, 1e-8, size=mesh_shape).astype(np.float64, order="F")

    # temperature [K]
    temp = np.full(mesh_shape, 1e4, dtype=np.float64, order="F")

    def create_triplets(
        r1: tuple[float, float],
        r2: tuple[float, float],
        r3: tuple[float, float],
        size: tuple[int, int, int] = mesh_shape,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal rng
        a = rng.uniform(*r1, size=size).astype(np.float64, order="F")
        b = rng.uniform(*r2, size=size).astype(np.float64, order="F")
        c = rng.uniform(*r3, size=size).astype(np.float64, order="F")
        return a, b, c

    # Ionization fractions
    xHIs = create_triplets((0, 1e-1), (0, 1e-1), (0, 1e-1))
    xHeIs = create_triplets((0, 1e-2), (0, 1e-2), (0, 1e-2))
    xHeIIs = create_triplets((0, 1e-3), (0, 1e-3), (0, 1e-3))

    # Photo-ionization rates
    phion = create_triplets((1e-13, 1e-12), (1e-14, 1e-13), (1e-15, 1e-14))
    pheat = create_triplets((1e-23, 1e-22), (1e-24, 1e-23), (1e-25, 1e-24))

    # Clumping factor
    clump = np.ones(mesh_shape, dtype=np.float64, order="F")

    yield (
        dt,
        dr,
        Hz,
        temp,
        temp,
        ndens,
        *xHIs,
        *xHeIs,
        *xHeIIs,
        *phion,
        *pheat,
        clump,
    )


@pytest.mark.skip(reason="still under R&D")
def test_chemistry(data_dir):
    with setup_chemistry() as args:
        xHI = args[6]
        xHI_int = args[8]
        xHeI = args[9]
        xHeI_int = args[11]
        xHeII = args[12]
        xHeII_int = args[14]

        for _ in range(10):
            conv = libc2ray.chemistry_he.global_pass(*args)
            xHI[:] = xHI_int
            xHeI[:] = xHeI_int
            xHeII[:] = xHeII_int

        # expected_xh = np.load(data_dir / "ionized_fraction_average.npy")
        assert conv == 0
        # assert np.allclose(xh, expected_xh)
