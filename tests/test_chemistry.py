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
    get_electron_density,
    get_energy,
    get_temperature,
    thermal,
)


def test_get_temperature_and_energy():
    temp = 1e4  # K
    ndens = 1e-5  # cm^-3
    energy = get_energy(temp, ndens)
    temp_calculated = get_temperature(energy, ndens)
    assert math.isclose(temp, temp_calculated)


def test_get_electron_density():
    n_a = 1.0
    xh = 0.3, 0.2, 0.1
    assert math.isclose(get_electron_density(n_a, xh, abu_h=1, abu_he=0, abu_c=0), 0.3)
    assert math.isclose(get_electron_density(n_a, xh, abu_h=0, abu_he=1, abu_c=0), 0.4)
    assert math.isclose(get_electron_density(n_a, xh, abu_h=1, abu_he=1, abu_c=0), 0.7)


def test_cooling_tables(data_dir) -> None:
    with pytest.raises(FileNotFoundError):
        CoolingTables.from_dir(data_dir)
    with pytest.raises(FileNotFoundError):
        CoolingTables.from_dir(data_dir / "non_existent_directory")

    cool_tables = CoolingTables.from_dir()
    assert "tables" in Path(cool_tables.tables_directory).parts
    assert "cooling" in Path(cool_tables.tables_directory).parts

    assert cool_tables.logtemp == (1.0, 0.01, 800)
    assert len(cool_tables.HI) == 801
    assert max(cool_tables.HI) < 1.0
    assert len(cool_tables.HII) == 801
    assert max(cool_tables.HII) < 1.0
    assert len(cool_tables.HeI) == 801
    assert max(cool_tables.HeI) < 1.0
    assert len(cool_tables.HeII) == 801
    assert max(cool_tables.HeII) < 1.0
    assert len(cool_tables.HeIII) == 801
    assert max(cool_tables.HeIII) < 1.0


@pytest.fixture(scope="module")
def cooling_tables() -> CoolingTables:
    return CoolingTables.from_dir()


def test_cooling_rate(cooling_tables: CoolingTables) -> None:
    abu_h = 0.76
    abu_he = 0.24
    xh = 0.9, 0.7, 0.2
    n_a = 1e-4
    n_e = get_electron_density(n_a, xh, abu_h=abu_h, abu_he=abu_he)
    cool_tables = CoolingTables.from_dir()

    rate = cooling_rate(n_a, n_e, 1e5, *xh, cool_tables, abu_h, abu_he)
    assert math.isclose(rate * 1e27, 1.09782637)


def test_cosmo_cooling_rate() -> None:
    temp = 1e4  # K
    ndens = 1e-5  # cm^-3
    energy = get_energy(temp, ndens)
    Hz = cosmo.H(10).cgs.value
    rate = cosmo_cooling_rate(energy, Hz)
    assert math.isclose(rate * 1e33, 1.852166392)


@pytest.mark.parametrize("n_a", (1e-6, 1e-5, 1e-4, 1e-3))
@pytest.mark.parametrize("temp", [10, 100, 1000, 10000])
def test_compare_cooling_rates(
    temp: float, n_a: float, cooling_tables: CoolingTables
) -> None:
    abu_h = 0.76
    abu_he = 0.24
    xh = 0.9, 0.7, 0.2
    n_e = get_electron_density(n_a, xh, abu_h=abu_h, abu_he=abu_he)
    cool_tables = CoolingTables.from_dir()
    cool_rate = cooling_rate(n_a, n_e, temp, *xh, cool_tables, abu_h, abu_he)

    Hz = cosmo.H(5).cgs.value
    cosmo_rate = cosmo_cooling_rate(get_energy(temp, n_a + n_e), Hz)
    assert 0.001 < cool_rate / cosmo_rate < 100.0


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
    """Testing thermal evolution solver. The analytical solution when just accounting
    for the expansion of the Universe should follow T0 * np.pow(times / times[0], -4 / 3)
    """

    expected_temps_cosmo_only = "thermal_evolution_cosmo_only.npz"
    expected_temps = "thermal_evolution.npz"
    ndens = 1e-5
    xh = 0.9, 0.7, 0.2

    def test_pyc2ray_cosmo_only(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times
        dts = np.diff(times)

        T0 = cosmo.Tcmb(zreds[0]).cgs.value

        temps = np.full_like(times, T0)
        temps_av = np.full_like(times, T0)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, TA = thermal(dt, T0, 1.0, 1.0, 0.0, Hz, cosmo_only=True)
            temps[i] = T0
            temps_av[i] = TA

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps_cosmo_only)
        assert np.allclose(temps, exp_temps["T0"])
        assert np.allclose(temps_av, exp_temps["TA"])

    def test_c2ray_cosmo_only(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times
        dts = np.diff(times)

        T_start = cosmo.Tcmb(zreds[0]).cgs.value
        T0 = np.array(T_start, dtype=np.float64)
        TA = T0.copy()

        temps = np.full_like(times, T0)
        temps_av = np.full_like(times, TA)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            libc2ray.chemistry_he.thermal(dt, T0, TA, 1.0, 1.0, 0.0, Hz, True)
            temps[i] = T0
            temps_av[i] = TA

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps_cosmo_only)
        assert np.allclose(temps, exp_temps["T0"])
        assert np.allclose(temps_av, exp_temps["TA"])

    def test_asora_cosmo_only(
        self, data_dir: Path, gen_times: tuple[np.ndarray, np.ndarray]
    ) -> None:
        times, zreds = gen_times
        dts = np.diff(times)

        T0 = cosmo.Tcmb(zreds[0]).cgs.value

        temps = np.full_like(times, T0)
        temps_av = np.full_like(times, T0)
        assert libasora_He is not None
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, TA = libasora_He.chemistry_thermal(
                dt, T0, 1.0, 1.0, 0.0, Hz, cosmo_only=True
            )
            temps[i] = T0
            temps_av[i] = TA

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps_cosmo_only)
        assert np.allclose(temps, exp_temps["T0"])
        assert np.allclose(temps_av, exp_temps["TA"])

    def test_pyc2ray_fail(self) -> None:
        with pytest.raises(ValueError):
            thermal(1, 1e4, 1.0, 1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            thermal(1, 1e4, 1.0, 1.0, 0.0, 1.0, (1e-7, 1e-8, 1e-9))

    def test_pyc2ray(
        self,
        data_dir: Path,
        gen_times: tuple[np.ndarray, np.ndarray],
        cooling_tables: CoolingTables,
    ) -> None:
        times, zreds = gen_times
        dts = np.diff(times)

        T0 = cosmo.Tcmb(zreds[0]).cgs.value
        n_e = get_electron_density(self.ndens, self.xh)

        temps = np.full_like(times, T0)
        temps_av = np.full_like(times, T0)
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, TA = thermal(dt, T0, n_e, self.ndens, 0.0, Hz, self.xh, cooling_tables)
            temps[i] = T0
            temps_av[i] = TA

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps)
        assert np.allclose(temps, exp_temps["T0"])
        assert np.allclose(temps_av, exp_temps["TA"])

    def test_asora(
        self,
        data_dir: Path,
        gen_times: tuple[np.ndarray, np.ndarray],
        cooling_tables: CoolingTables,
    ) -> None:
        times, zreds = gen_times
        dts = np.diff(times)

        T0 = cosmo.Tcmb(zreds[0]).cgs.value
        n_e = get_electron_density(self.ndens, self.xh)

        temps = np.full_like(times, T0)
        temps_av = np.full_like(times, T0)
        assert libasora_He is not None
        for i, (zi, dt) in enumerate(zip(zreds[:-1], dts), 1):
            Hz = cosmo.H(zi).cgs.value
            T0, TA = libasora_He.chemistry_thermal(
                dt,
                T0,
                n_e,
                self.ndens,
                0.0,
                Hz,
                *self.xh,
                *cooling_tables.astuple(),
                *cooling_tables.logtemp,
            )
            temps[i] = T0
            temps_av[i] = TA

        exp_temps = np.load(data_dir / TestThermalEvolution.expected_temps)
        assert np.allclose(temps, exp_temps["T0"])
        assert np.allclose(temps_av, exp_temps["TA"])


@contextmanager
def setup_chemistry(
    mesh_size: int = 10,
    ionize_species: tuple[bool, bool, bool] = (True, True, True),
    cosmo_only: bool = False,
    heat: float = 1e-31,
):
    assert len(ionize_species) == 3, "ionize_species should be a tuple of 3 booleans"

    mesh_shape = (mesh_size,) * 3
    rng = np.random.default_rng(2023)

    dt = (5 * u.Myr).cgs.value
    Hz = cosmo.H(10).cgs.value

    # density field [g/cm^3]
    ndens = rng.uniform(1e-6, 1e-3, size=mesh_shape).astype(np.float64, order="F")

    # temperature [K]
    temp = np.pow(10, rng.normal(4, 0.25, size=mesh_shape), dtype=np.float64, order="F")

    # Ionization fractions for x, x_av and x_int
    xHIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    xHeIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    xHeIIIs = tuple(np.zeros_like(ndens) for _ in ionize_species)
    phion = tuple(
        np.full_like(ndens, s * p)
        for s, p in zip(ionize_species, (1e-14, 1e-15, 1e-16))
    )
    pheat = np.full_like(ndens, heat)

    # Clumping factor
    clump = np.ones_like(ndens)

    logtemp: tuple
    if not cosmo_only:
        assert libasora_He is not None
        cool_tables = CoolingTables.from_dir()
        libasora_He.cooling_tables_to_device(
            cool_tables.HI,
            cool_tables.HII,
            cool_tables.HeI,
            cool_tables.HeII,
            cool_tables.HeIII,
        )
        logtemp = cool_tables.logtemp
    else:
        logtemp = tuple()

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
        pheat,
        clump,
        cosmo_only,
        *logtemp,
    )


def compare_ionized_fractions(results_file: Path, args: tuple) -> None:
    exp_xh = np.load(results_file)
    for idx, name in {
        7: "xHII",
        10: "xHeII",
        13: "xHeIII",
        6: "xHII_av",
        9: "xHeII_av",
        12: "xHeIII_av",
    }.items():
        assert np.allclose(args[idx], exp_xh[name])


def test_chemistry_c2ray_no_heat_hydrogen_only_cosmo_only(data_dir):
    with setup_chemistry(
        heat=0.0, ionize_species=(True, False, False), cosmo_only=True
    ) as args:
        pheat_HeI = np.zeros_like(args[17])
        pheat_HeII = np.zeros_like(args[17])
        c2ray_args = *args[:18], pheat_HeI, pheat_HeII, *args[18:]

        conv = libc2ray.chemistry_he.global_pass(*c2ray_args)
        assert conv == 0
        compare_ionized_fractions(
            data_dir / "ionized_fraction_no_heat_hydrogen_only_cosmo_only.npz", args
        )


def test_chemistry_c2ray_no_heat_cosmo_only(data_dir):
    with setup_chemistry(heat=0.0, cosmo_only=True) as args:
        pheat_HeI = np.zeros_like(args[17])
        pheat_HeII = np.zeros_like(args[17])
        c2ray_args = *args[:18], pheat_HeI, pheat_HeII, *args[18:]

        conv = libc2ray.chemistry_he.global_pass(*c2ray_args)
        assert conv == 0
        compare_ionized_fractions(
            data_dir / "ionized_fraction_no_heat_all_species_cosmo_only.npz", args
        )


def test_chemistry_asora_no_heat_hydrogen_only_cosmo_only(data_dir, init_device):
    with setup_chemistry(
        heat=0.0, ionize_species=(True, False, False), cosmo_only=True
    ) as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        conv = libasora_He.chemistry_global_pass(*asora_args)

        # Solution doesn't converge in one step.
        assert conv == 1000
        compare_ionized_fractions(
            data_dir / "ionized_fraction_no_heat_hydrogen_only_cosmo_only.npz", args
        )


def test_chemistry_asora_no_heat_cosmo_only(data_dir, init_device):
    with setup_chemistry(heat=0.0, cosmo_only=True) as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        conv = libasora_He.chemistry_global_pass(*asora_args)

        # Solution doesn't converge in one step.
        assert conv == 1000
        compare_ionized_fractions(
            data_dir / "ionized_fraction_no_heat_all_species_cosmo_only.npz", args
        )


def test_chemistry_asora_no_heat_hydrogen_only(data_dir, init_device):
    with setup_chemistry(heat=0.0, ionize_species=(True, False, False)) as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        conv = libasora_He.chemistry_global_pass(*asora_args)

        # Solution doesn't converge in one step.
        assert conv == 1000
        compare_ionized_fractions(
            data_dir / "ionized_fraction_no_heat_hydrogen_only.npz", args
        )


def test_chemistry_asora_no_heat(data_dir, init_device):
    with setup_chemistry(heat=0.0) as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        conv = libasora_He.chemistry_global_pass(*asora_args)

        # Solution doesn't converge in one step.
        assert conv == 1000
        compare_ionized_fractions(
            data_dir / "ionized_fraction_no_heat_all_species.npz", args
        )


def test_chemistry_asora_hydrogen_only(data_dir, init_device):
    with setup_chemistry(ionize_species=(True, False, False), heat=0.0) as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        conv = libasora_He.chemistry_global_pass(*asora_args)

        # Solution doesn't converge in one step.
        assert conv == 1000
        compare_ionized_fractions(data_dir / "ionized_fraction_hydrogen_only.npz", args)


def test_chemistry_asora(data_dir, init_device):
    with setup_chemistry() as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        conv = libasora_He.chemistry_global_pass(*asora_args)

        # Solution doesn't converge in one step.
        assert conv == 1000
        compare_ionized_fractions(data_dir / "ionized_fraction_all_species.npz", args)


### BENCHMARKS ###


def test_benchmark_chemistry_c2ray(benchmark):
    with setup_chemistry(100, cosmo_only=True) as args:
        benchmark(libc2ray.chemistry_he.global_pass, *args)


@pytest.mark.parametrize("block_size", [128, 256, 512])
def test_benchmark_chemistry_asora(benchmark, init_device, block_size):
    with setup_chemistry(100) as args:
        assert libasora_He is not None
        libasora_He.density_to_device(args[2])

        asora_args = args[:2] + args[3:]
        benchmark(libasora_He.chemistry_global_pass, *asora_args, block_size)
