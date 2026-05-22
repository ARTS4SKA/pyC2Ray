import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from pyc2ray.load_extensions import libasora_He, libc2ray
from pyc2ray.solver.helium import (
    CoolingTables,
    cooling_rate,
    cosmo_cooling_rate,
    get_energy,
    get_temperature,
    thermal,
)


def test_get_temperature_and_energy():
    temp = 1e4  # K
    ndens = 1e-7  # cm^-3
    energy = get_energy(temp, ndens)
    temp_calculated = get_temperature(energy, ndens)
    assert math.isclose(temp, temp_calculated)


def test_cooling_tables(data_dir) -> None:
    with pytest.raises(FileNotFoundError):
        CoolingTables.from_dir(data_dir)
    with pytest.raises(FileNotFoundError):
        CoolingTables.from_dir(data_dir / "non_existent_directory")

    cool_tables = CoolingTables.from_dir()
    assert "tables" in Path(cool_tables.tables_directory).parts
    assert "cooling" in Path(cool_tables.tables_directory).parts

    assert cool_tables.logtemp == (1.0, 0.01, 801)
    assert len(cool_tables.HI) == 802
    assert max(cool_tables.HI) < 1.0
    assert len(cool_tables.HII) == 802
    assert max(cool_tables.HII) < 1.0
    assert len(cool_tables.HeI) == 802
    assert max(cool_tables.HeI) < 1.0
    assert len(cool_tables.HeII) == 802
    assert max(cool_tables.HeII) < 1.0
    assert len(cool_tables.HeIII) == 802
    assert max(cool_tables.HeIII) < 1.0


def test_cooling_rate() -> None:
    xHI, xHeI, xHeII = 0.9, 0.7, 0.2
    n_a = 1e-8
    n_e = n_a * ((1 - xHI) + xHeII + 2 * (1 - xHeI - xHeII))
    cool_tables = CoolingTables.from_dir()

    rate = cooling_rate(
        n_a, n_e, 1e5, xHI, xHeI, xHeII, cool_tables, abu_h=0.76, abu_he=0.24
    )
    assert math.isclose(rate * 1e36, 5.79021913)


def test_cosmo_cooling_rate() -> None:
    assert math.isclose(cosmo_cooling_rate(0.2, 0.1), 0.04)


@pytest.fixture(scope="class")
def gen_times(
    zi: float = 40, zf: float = 2, nt: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    ti = cosmo.age(zi).cgs.value
    tf = cosmo.age(zf).cgs.value
    times = np.linspace(ti, tf, nt, endpoint=False)
    zreds = z_at_value(cosmo.age, times * u.s).value
    return times, zreds


class TestThermalEvolution:
    expected_temps_cosmo_only = "thermal_evolution_cosmo_only.npy"
    expected_temps = "thermal_evolution.npy"

    def test_pyc2ray_cosmo_only(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times

        T0 = cosmo.Tcmb(zreds[0]).cgs.value

        # Get analytical solution for the temperature evolution
        temps_anal = T0 * np.pow(times / times[0], -4 / 3)

        # Get numerical solution using the thermal function
        dts = np.diff(times)
        temps_num = np.full_like(temps_anal, T0)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, _ = thermal(dt, T0, 1.0, 1.0, 0.0, Hz, cosmo_only=True)
            temps_num[i] = T0

        # TODO: change this tests once we can update H during the evolution.
        assert (np.abs(temps_num - temps_anal) / temps_anal < 0.75).all()

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps_cosmo_only)
        assert np.allclose(temps_num, exp_temps)

    def test_c2ray_cosmo_only(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times

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
            libc2ray.chemistry_he.thermal(dt, T0, TA, 1.0, 1.0, 0.0, Hz, True)
            temps_num[i] = T0

        # TODO: change this tests once we can update H during the evolution.
        assert (np.abs(temps_num - temps_anal) / temps_anal < 0.75).all()

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps_cosmo_only)
        assert np.allclose(temps_num, exp_temps)

    def test_asora_cosmo_only(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times

        T_start = cosmo.Tcmb(zreds[0]).cgs.value
        T0 = np.array(T_start, dtype=np.float64)

        # Get analytical solution for the temperature evolution
        temps_anal = T0 * np.pow(times / times[0], -4 / 3)

        # Get numerical solution using the thermal function
        dts = np.diff(times)
        temps_num = np.full_like(temps_anal, T0)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            assert libasora_He is not None
            T0, _ = libasora_He.chemistry_thermal(
                dt, T0, 1.0, 1.0, 0.0, Hz, cosmo_only=True
            )
            temps_num[i] = T0

        # TODO: change this tests once we can update H during the evolution.
        assert (np.abs(temps_num - temps_anal) / temps_anal < 0.75).all()

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps_cosmo_only)
        assert np.allclose(temps_num, exp_temps)

    def test_pyc2ray_fail(self) -> None:
        with pytest.raises(ValueError):
            thermal(1, 1e4, 1.0, 1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            thermal(1, 1e4, 1.0, 1.0, 0.0, 1.0, (1e-7, 1e-8, 1e-9))

    def test_pyc2ray(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times
        T0 = cosmo.Tcmb(zreds[0]).cgs.value

        cool_tables = CoolingTables.from_dir()
        xh = (0.9, 0.7, 0.2)
        n_a = 1e-8
        n_e = n_a * ((1 - xh[0]) + xh[2] + 2 * (1 - xh[1] - xh[2]))

        # Get numerical solution using the thermal function
        dts = np.diff(times)
        temps = np.full_like(times, T0)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, _ = thermal(dt, T0, n_e, n_a, 0.0, Hz, xh, cool_tables)
            temps[i] = T0

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps)
        assert np.allclose(temps, exp_temps)

    def test_asora(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times
        T0 = cosmo.Tcmb(zreds[0]).cgs.value

        tables = CoolingTables.from_dir()
        cool_tables = tables.HI, tables.HII, tables.HeI, tables.HeII, tables.HeIII
        xh = (0.9, 0.7, 0.2)
        n_a = 1e-8
        n_e = n_a * ((1 - xh[0]) + xh[2] + 2 * (1 - xh[1] - xh[2]))

        # Get numerical solution using the thermal function
        dts = np.diff(times)
        temps = np.full_like(times, T0)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, _ = libasora_He.chemistry_thermal(
                dt, T0, n_e, n_a, 0.0, Hz, *xh, *cool_tables, *tables.logtemp
            )
            temps[i] = T0

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps)
        assert np.allclose(temps, exp_temps)


@contextmanager
def setup_chemistry(
    mesh_size: int = 10,
    ionize_species: tuple[bool, bool, bool] = (True, True, True),
    cosmo_only: bool = False,
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
        cosmo_only,
    )


def test_chemistry_c2ray_hydrogen_only_cosmo_only(data_dir):
    with setup_chemistry(ionize_species=(True, False, False), cosmo_only=True) as args:
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

        expected_xHII = np.load(
            data_dir / "ionized_fraction_hydrogen_only_cosmo_only.npy"
        )
        assert np.allclose(xHII, expected_xHII)


def test_chemistry_c2ray_cosmo_only(data_dir):
    with setup_chemistry(cosmo_only=True) as args:
        xHII, xHII_av, xHII_int = args[5:8]
        xHeII, xHeII_av, xHeII_int = args[8:11]
        xHeIII, xHeIII_av, xHeIII_int = args[11:14]

        for s in range(10):
            conv = libc2ray.chemistry_he.global_pass(*args)
            xHII[:] = xHII_int
            xHeII[:] = xHeII_int
            xHeIII[:] = xHeIII_int

        assert conv == 0

        exp_xh = np.load(data_dir / "ionized_fraction_all_species_cosmo_only.npz")
        assert np.allclose(xHII, exp_xh["xHII"])
        assert np.allclose(xHeII, exp_xh["xHeII"])
        assert np.allclose(xHeIII, exp_xh["xHeIII"])
        assert np.allclose(xHII_av, exp_xh["xHII_av"])
        assert np.allclose(xHeII_av, exp_xh["xHeII_av"])
        assert np.allclose(xHeIII_av, exp_xh["xHeIII_av"])


def test_chemistry_asora_hydrogen_only_cosmo_only(data_dir, init_device):
    with setup_chemistry(ionize_species=(True, False, False), cosmo_only=True) as args:
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

        expected_xHII = np.load(
            data_dir / "ionized_fraction_hydrogen_only_cosmo_only.npy"
        )
        assert np.allclose(xHII, expected_xHII)


def test_chemistry_asora_cosmo_only(data_dir, init_device):
    with setup_chemistry(cosmo_only=True) as args:
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

        exp_xh = np.load(data_dir / "ionized_fraction_all_species_cosmo_only.npz")
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
