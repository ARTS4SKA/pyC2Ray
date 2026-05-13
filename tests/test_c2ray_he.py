from contextlib import contextmanager

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from pyc2ray.load_extensions import libasora_He, libc2ray
from pyc2ray.solver.helium import thermal


def test_load_c2ray() -> None:
    assert libc2ray is not None
    assert hasattr(libc2ray, "chemistry_he")
    assert hasattr(libc2ray.chemistry_he, "thermal")


def test_thermal_evolution_only_cosmic_expansion_python(data_dir) -> None:
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

    expected_temps = np.load(data_dir / "thermal_cosmic_evolution.npy")
    assert np.allclose(temps_num, expected_temps)


def test_thermal_evolution_only_cosmic_expansion_fortran(data_dir) -> None:
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

    expected_temps = np.load(data_dir / "thermal_cosmic_evolution.npy")
    assert np.allclose(temps_num, expected_temps)


def test_thermal_evolution_only_cosmic_expansion_cuda(data_dir) -> None:
    ti = cosmo.age(40).cgs.value
    tf = cosmo.age(2).cgs.value
    times = np.linspace(ti, tf, 200, endpoint=False)
    zreds = z_at_value(cosmo.age, times * u.s).value

    T_start = cosmo.Tcmb(zreds[0]).cgs.value
    T0 = np.array(T_start, dtype=np.float64)

    # Get analytical solution for the temperature evolution
    temps_anal = T0 * np.pow(times / times[0], -4 / 3)

    # Get numerical solution using the thermal function
    dts = np.diff(times)
    temps_num = np.full_like(temps_anal, T0)
    for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
        Hz = cosmo.H(zi).cgs.value
        # These combinations of parameters disables heating and cooling rates:
        # the temperature evolution is only due to cosmic expansion
        assert libasora_He is not None
        T0, _ = libasora_He.chemistry_thermal(dt, T0, 1.0, 0.0, 0.0, Hz)
        temps_num[i] = T0

    # TODO: change this tests once we can update H during the evolution.
    assert (np.abs(temps_num - temps_anal) / temps_anal < 0.75).all()

    expected_temps = np.load(data_dir / "thermal_cosmic_evolution.npy")
    assert np.allclose(temps_num, expected_temps)


@contextmanager
def setup_chemistry(
    mesh_size: int = 10, ionize_species: tuple[bool, bool, bool] = (True, True, True)
):
    assert len(ionize_species) == 3, "ionize_species should be a tuple of 3 booleans"

    mesh_shape = (mesh_size,) * 3
    rng = np.random.default_rng(2023)

    dt = (1 * u.Myr).cgs.value
    Hz = cosmo.H(10).cgs.value

    # density field [g/cm^3]
    ndens = rng.uniform(5e-8, 5e-7, size=mesh_shape).astype(np.float64, order="F")

    # temperature [K]
    temp = rng.normal(1e4, 10, size=mesh_shape).astype(np.float64, order="F")

    # Ionization fractions for x, x_av and x_int
    xHIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    xHeIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    xHeIIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    phion = tuple(
        np.full_like(ndens, s * p)
        for s, p in zip(ionize_species, (1e-14, 1e-15, 1e-16))
    )
    # Unused right now
    pheat = tuple(np.zeros_like(ndens) for _ in ionize_species)

    # Clumping factor
    clump = np.ones_like(ndens)

    yield (
        dt,
        Hz,
        ndens,
        temp,
        temp,
        *xHIIs,
        *xHeIIs,
        *xHeIIIs,
        *phion,
        *pheat,
        clump,
    )


def test_chemistry_hydrogen_only(data_dir):
    with setup_chemistry(ionize_species=(True, False, False)) as args:
        xHII, xHII_av, xHII_int = args[5:8]
        xHeII, xHeII_av, xHeII_int = args[8:11]
        xHeIII, xHeIII_av, xHeIII_int = args[11:14]

        for s in range(10):
            conv = libc2ray.chemistry_he.global_pass(*args)
            xHII[:] = xHII_int
            xHeII[:] = xHeII_int
            xHeIII[:] = xHeIII_int
            assert np.allclose(xHeII, 0.0)
            assert np.allclose(xHeII_av, 0.0)
            assert np.allclose(xHeIII, 0.0)
            assert np.allclose(xHeIII_av, 0.0)

        assert conv == 0

        expected_xHII = np.load(data_dir / "ionized_fraction_only_hydrogen.npy")
        assert np.allclose(xHII, expected_xHII)


@pytest.fixture
def init_device():
    libasora_He.device_init()
    yield
    libasora_He.device_close()


def test_chemistry_hydrogen_only_asora(data_dir, init_device):
    with setup_chemistry(ionize_species=(True, False, False)) as args:
        xHII, xHII_av, xHII_int = args[5:8]
        xHeII, xHeII_av, xHeII_int = args[8:11]
        xHeIII, xHeIII_av, xHeIII_int = args[11:14]

        assert libasora_He is not None
        for s in range(10):
            conv = libasora_He.chemistry_global_pass(*args)
            xHII[:] = xHII_int
            xHeII[:] = xHeII_int
            xHeIII[:] = xHeIII_int

            assert np.allclose(xHeII, 0.0)
            assert np.allclose(xHeII_av, 0.0)
            assert np.allclose(xHeIII, 0.0)
            assert np.allclose(xHeIII_av, 0.0)

        assert conv == 0

        expected_xHII = np.load(data_dir / "ionized_fraction_only_hydrogen.npy")
        assert np.allclose(xHII, expected_xHII)


def test_chemistry(data_dir):
    with setup_chemistry() as args:
        xHII, xHII_av, xHII_int = args[5:8]
        xHeII, xHeII_av, xHeII_int = args[8:11]
        xHeIII, xHeIII_av, xHeIII_int = args[11:14]

        for s in range(10):
            conv = libc2ray.chemistry_he.global_pass(*args)
            xHII[:] = xHII_int
            xHeII[:] = xHeII_int
            xHeIII[:] = xHeIII_int

        assert conv == 0

        exp_xh = np.load(data_dir / "ionized_fraction_all_species.npz")
        assert np.allclose(xHII, exp_xh["xHII"])
        assert np.allclose(xHeII, exp_xh["xHeII"])
        assert np.allclose(xHeIII, exp_xh["xHeIII"])
        assert np.allclose(xHII_av, exp_xh["xHII_av"])
        assert np.allclose(xHeII_av, exp_xh["xHeII_av"])
        assert np.allclose(xHeIII_av, exp_xh["xHeIII_av"])


def test_chemistry_asora(data_dir, init_device):
    with setup_chemistry() as args:
        xHII, xHII_av, xHII_int = args[5:8]
        xHeII, xHeII_av, xHeII_int = args[8:11]
        xHeIII, xHeIII_av, xHeIII_int = args[11:14]

        assert libasora_He is not None
        for s in range(10):
            conv = libasora_He.chemistry_global_pass(*args)
            xHII[:] = xHII_int
            xHeII[:] = xHeII_int
            xHeIII[:] = xHeIII_int

        assert conv == 0

        # TODO: why must the tolerance be so high?
        exp_xh = np.load(data_dir / "ionized_fraction_all_species.npz")
        assert np.allclose(xHII, exp_xh["xHII"], rtol=1e-5)
        assert np.allclose(xHeII, exp_xh["xHeII"], rtol=1e-4)
        assert np.allclose(xHeIII, exp_xh["xHeIII"], rtol=1e-3)
        assert np.allclose(xHII_av, exp_xh["xHII_av"], rtol=1e-5)
        assert np.allclose(xHeII_av, exp_xh["xHeII_av"], rtol=1e-4)
        assert np.allclose(xHeIII_av, exp_xh["xHeIII_av"], rtol=1e-3)


def test_benchmark_chemistry(benchmark):
    with setup_chemistry(200) as args:
        benchmark(libc2ray.chemistry_he.global_pass, *args)


@pytest.mark.parametrize("block_size", [128, 256, 512])
def test_benchmark_chemistry_asora(benchmark, init_device, block_size):
    with setup_chemistry(200) as args:
        benchmark(libasora_He.chemistry_global_pass, *args, block_size)
