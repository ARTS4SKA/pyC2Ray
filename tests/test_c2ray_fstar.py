"""Characterization tests for pyc2ray.c2ray_fstar.C2Ray_fstar

The C2Ray base class is not initialised: instances are created with __new__ and
only the attributes used by the methods under test are set.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias
from unittest import mock

import h5py
import numpy as np
import pytest

import pyc2ray.constants as c
from pyc2ray import c2ray_fstar
from pyc2ray.c2ray_fstar import C2Ray_fstar
from pyc2ray.parameters import (
    CosmologyParameters,
    GridParameters,
    InputOutputParameters,
    MaterialParameters,
    SourcesParameters,
)
from pyc2ray.utils import bin_sources
from pyc2ray.utils.sourceutils import FloatArray

TestCase: TypeAlias = tuple[str, str, str, str]
Haloes: TypeAlias = tuple[FloatArray, FloatArray]
FstarFactory: TypeAlias = Callable[..., C2Ray_fstar]

H = 0.7
S_STAR_REF = 1e48
BOXSIZE = 10.0  # Mpc/h
MESHSIZE = 8
NUM_HALOES = 64
REDSHIFT = 8.0
MEAN_MOLECULAR = 1.22
DT = 10.0 * 1e6 * c.year2s  # 10 Myr in s
TS = 50.0 * 1e6 * c.year2s  # patched source lifetime


SOURCE_DEFAULTS: dict[str, Any] = {
    "fstar_model": "fgamma",
    "fesc_model": "constant",
    "accretion_model": "constant",
    "bursty_sfr": "no",
    "Nion": 2000.0,
    "f0": 0.02,
    "Mt": 1e8,
    "Mp": 1e10,
    "g1": -0.5,
    "g2": 0.5,
    "g3": 3.0,
    "g4": -1.0,
    "a_s": -0.20046,
    "b_s": 3.34619,
    "alpha_h": 0.79,
    "f0_esc": 0.1,
    "Mp_esc": 1e10,
    "al_esc": -0.3,
    "beta1": 0.1,
    "beta2": 1.5,
    "tB0": 200.0,
    "tQ_frac": 1.6,
    "z0": 30.0,
    "t_rnd": 0,
    "seed": 918273645,
}

# =======
# Helpers
# =======


def make_haloes() -> Haloes:
    """Synthetic halo catalogue: positions [Mpc] inside the box and masses [Msun]"""
    rng = np.random.default_rng(1234)
    pos = rng.uniform(0.0, BOXSIZE / H, size=(NUM_HALOES, 3))
    mass = 10 ** rng.uniform(8.0, 12.0, size=NUM_HALOES)
    return pos, mass


def build_fstar(path: Path = Path("."), **overrides: Any) -> C2Ray_fstar:
    """Create a C2Ray_fstar without running the C2Ray constructor"""
    sim: Any = C2Ray_fstar.__new__(C2Ray_fstar)
    sim.parameters = SimpleNamespace(
        sources=SourcesParameters.from_dict(SOURCE_DEFAULTS | overrides),
        grid=GridParameters.from_dict(
            {
                "boxsize": BOXSIZE,
                "meshsize": MESHSIZE,
                "gpu": False,
                "mpi": False,
                "resume": False,
            }
        ),
        io=InputOutputParameters.from_dict(
            {
                "results_dir": path,
                "sources_dir": path,
                "density_dir": path,
            }
        ),
        material=MaterialParameters.from_dict({}),
        cosmology=CosmologyParameters.from_dict({"h": H}),
    )
    sim._cosmology_init()
    return sim


# TODO: refactor once this is not needed outside
def run_test_case(case: TestCase) -> dict[str, Any]:
    fstar_model, fesc_model, accretion_model, bursty_sfr = case
    sim = build_fstar(
        fstar_model=fstar_model,
        fesc_model=fesc_model.lower(),
        accretion_model=accretion_model.lower(),
        bursty_sfr=bursty_sfr.lower(),
    )
    haloes = make_haloes()
    sim.read_haloes = lambda *_: haloes  # type: ignore[method-assign,assignment]

    sim._sources_init()
    srcpos, normflux = sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

    return {
        "srcpos": srcpos,
        "normflux": normflux,
        "tot_phots": np.float64(sim.tot_phots),
        "perc_switchon": np.float64(sim.perc_switchon),
    }


@pytest.fixture
def make_fstar(tmp_path: Path) -> FstarFactory:
    def factory(**overrides: Any) -> C2Ray_fstar:
        return build_fstar(tmp_path, **overrides)

    return factory


# ======================================================================
# _sources_init
# ======================================================================


@pytest.fixture
def patched_models(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    models = SimpleNamespace(
        fstar=mock.MagicMock(), fesc=mock.MagicMock(), bursty=mock.MagicMock()
    )
    monkeypatch.setattr(c2ray_fstar, "StellarToHaloRelation", models.fstar)
    monkeypatch.setattr(c2ray_fstar, "EscapeFraction", models.fesc)
    monkeypatch.setattr(c2ray_fstar, "BurstySFR", models.bursty)
    return models


class TestSourcesInit:
    @pytest.mark.parametrize("fstar_model", ["fgamma", "dpl", "Muv"])
    def test_fstar_model_built(
        self,
        make_fstar: FstarFactory,
        patched_models: SimpleNamespace,
        fstar_model: str,
    ) -> None:
        sim = make_fstar(fstar_model=fstar_model)
        sim._sources_init()

        patched_models.fstar.assert_called_once_with(
            sim.parameters.sources.fstar,
            cosmo=sim.cosmology,
            rng=mock.ANY,
        )
        assert sim.fstar_relation is patched_models.fstar.return_value

    @pytest.mark.parametrize("fesc_model", ["constant", "power", "power_obs", "thesan"])
    def test_fesc_model_built(
        self,
        make_fstar: FstarFactory,
        patched_models: SimpleNamespace,
        fesc_model: str,
    ) -> None:
        sim = make_fstar(fesc_model=fesc_model)
        sim._sources_init()

        patched_models.fesc.assert_called_once_with(sim.parameters.sources.fesc)
        assert sim.fesc_relation is patched_models.fesc.return_value

    @pytest.mark.parametrize("bursty_sfr", ["instant", "integrate"])
    def test_bursty_model_built(
        self,
        make_fstar: FstarFactory,
        patched_models: SimpleNamespace,
        bursty_sfr: str,
    ) -> None:
        sim = make_fstar(accretion_model="exp", bursty_sfr=bursty_sfr)
        sim._sources_init()

        patched_models.bursty.assert_called_once_with(
            sim.parameters.sources.accretion,
            cosmo=sim.cosmology,
            rng=mock.ANY,
        )
        assert sim.bursty_relation is patched_models.bursty.return_value

    def test_models_share_seeded_rng(
        self, make_fstar: FstarFactory, patched_models: SimpleNamespace
    ) -> None:
        sim = make_fstar(accretion_model="exp", bursty_sfr="instant")
        sim._sources_init()

        rng = patched_models.fstar.call_args.kwargs["rng"]
        assert patched_models.bursty.call_args.kwargs["rng"] is rng
        expected = np.random.default_rng(SOURCE_DEFAULTS["seed"])
        assert rng.random() == expected.random()

    def test_no_bursty_model(
        self, make_fstar: FstarFactory, patched_models: SimpleNamespace
    ) -> None:
        sim = make_fstar(bursty_sfr="no")
        sim._sources_init()

        patched_models.bursty.assert_not_called()
        assert not hasattr(sim, "bursty_relation")


# ======================================================================
# ionizing_flux: control flow
# ======================================================================


@pytest.fixture(scope="class")
def haloes() -> Haloes:
    return make_haloes()


@pytest.fixture
def make_fstar_with_haloes(
    make_fstar: FstarFactory, haloes: Haloes, monkeypatch: pytest.MonkeyPatch
) -> FstarFactory:
    """C2Ray_fstar with real source models reading synthetic haloes"""

    def factory(**overrides: Any) -> C2Ray_fstar:
        sim = make_fstar(**overrides)
        monkeypatch.setattr(sim, "read_haloes", lambda *_: haloes)
        sim._sources_init()
        return sim

    return factory


@pytest.fixture
def mock_bin_sources(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Wrap bin_sources to record its arguments"""
    mocked = mock.MagicMock(wraps=bin_sources)
    monkeypatch.setattr(c2ray_fstar, "bin_sources", mocked)
    return mocked


def expected_normflux(srcmstar: FloatArray, ts: float) -> FloatArray:
    return c.msun2g * SOURCE_DEFAULTS["Nion"] * srcmstar / (c.m_p * ts * S_STAR_REF)


class TestLifetime:
    def test_exp_uses_source_lifetime(
        self, make_fstar_with_haloes: FstarFactory, mock_bin_sources: mock.MagicMock
    ) -> None:
        sim = make_fstar_with_haloes(accretion_model="exp")
        with mock.patch.object(
            sim.fstar_relation, "source_lifetime", return_value=TS
        ) as source_lifetime:
            _, normflux = sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        source_lifetime.assert_called_once_with(REDSHIFT, SOURCE_DEFAULTS["alpha_h"])

        # Call bin_sources with the same arguments to get the expected value
        _, srcmstar = bin_sources(**mock_bin_sources.call_args.kwargs)
        np.testing.assert_allclose(normflux, expected_normflux(srcmstar, TS))

    def test_constant_uses_dt(
        self, make_fstar_with_haloes: FstarFactory, mock_bin_sources: mock.MagicMock
    ) -> None:
        sim = make_fstar_with_haloes(accretion_model="constant")
        with mock.patch.object(
            sim.fstar_relation, "source_lifetime"
        ) as source_lifetime:
            _, normflux = sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        source_lifetime.assert_not_called()
        # Call bin_sources with the same arguments to get the expected value
        _, srcmstar = bin_sources(**mock_bin_sources.call_args.kwargs)
        np.testing.assert_allclose(normflux, expected_normflux(srcmstar, DT))

    def test_constant_without_dt_raises(
        self, make_fstar_with_haloes: FstarFactory
    ) -> None:
        sim = make_fstar_with_haloes(accretion_model="constant")
        with pytest.raises(AssertionError):
            sim.ionizing_flux("haloes.txt", z=REDSHIFT)


class TestFstarDispatch:
    def test_muv_model(
        self, make_fstar_with_haloes: FstarFactory, haloes: Haloes
    ) -> None:
        sim = make_fstar_with_haloes(fstar_model="Muv", accretion_model="exp")
        with mock.patch.object(
            sim.fstar_relation, "get", wraps=sim.fstar_relation.get
        ) as get:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        get.assert_called_once()
        kwargs = get.call_args.kwargs
        assert kwargs.keys() == {"Mhalo", "z", "alpha_h", "a_s", "b_s"}
        np.testing.assert_array_equal(kwargs["Mhalo"], haloes[1])
        assert kwargs["z"] == REDSHIFT
        assert kwargs["a_s"] == SOURCE_DEFAULTS["a_s"]
        assert kwargs["b_s"] == SOURCE_DEFAULTS["b_s"]

    @pytest.mark.parametrize("fstar_model", ["fgamma", "dpl"])
    def test_non_muv_model(
        self, make_fstar_with_haloes: FstarFactory, haloes: Haloes, fstar_model: str
    ) -> None:
        sim = make_fstar_with_haloes(fstar_model=fstar_model)
        with mock.patch.object(
            sim.fstar_relation, "get", wraps=sim.fstar_relation.get
        ) as get:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        get.assert_called_once()
        kwargs = get.call_args.kwargs
        assert kwargs.keys() == {"Mhalo"}
        np.testing.assert_array_equal(kwargs["Mhalo"], haloes[1])


# Default is fstar_model="fgamma"
class TestFescDispatch:
    def test_fesc_constant(
        self,
        make_fstar_with_haloes: FstarFactory,
        haloes: Haloes,
        mock_bin_sources: mock.MagicMock,
    ) -> None:
        sim = make_fstar_with_haloes(fesc_model="constant")
        with mock.patch.object(sim.fesc_relation, "get") as get:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        get.assert_not_called()
        fstar = sim.fstar_relation.get(Mhalo=haloes[1])
        mstar = mock_bin_sources.call_args.kwargs["mstar_msun"]
        np.testing.assert_allclose(mstar, sim.fesc_relation.f0_esc * fstar * haloes[1])

    def test_fesc_power(
        self, make_fstar_with_haloes: FstarFactory, haloes: Haloes
    ) -> None:
        sim = make_fstar_with_haloes(fesc_model="power")
        with mock.patch.object(
            sim.fesc_relation, "get", wraps=sim.fesc_relation.get
        ) as get:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        kwargs = get.call_args.kwargs
        assert kwargs.keys() == {"Mhalo"}
        np.testing.assert_array_equal(kwargs["Mhalo"], haloes[1])

    def test_fesc_power_obs(
        self, make_fstar_with_haloes: FstarFactory, haloes: Haloes
    ) -> None:
        sim = make_fstar_with_haloes(fesc_model="power_obs")
        with mock.patch.object(
            sim.fesc_relation, "get", wraps=sim.fesc_relation.get
        ) as get:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        kwargs = get.call_args.kwargs
        assert kwargs.keys() == {"Mhalo"}
        fstar = sim.fstar_relation.get(Mhalo=haloes[1])
        np.testing.assert_allclose(kwargs["Mhalo"], fstar * haloes[1])

    def test_fesc_thesan(
        self, make_fstar_with_haloes: FstarFactory, haloes: Haloes
    ) -> None:
        sim = make_fstar_with_haloes(fesc_model="thesan")
        with mock.patch.object(
            sim.fesc_relation, "get", wraps=sim.fesc_relation.get
        ) as get:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        kwargs = get.call_args.kwargs
        assert kwargs.keys() == {"Mhalo", "z"}
        np.testing.assert_array_equal(kwargs["Mhalo"], haloes[1])
        assert kwargs["z"] == REDSHIFT


# Default is fstar_model="fgamma"
class TestBurstySFR:
    MASK = np.arange(NUM_HALOES) % 2 == 0

    # This combionation applies mask to fstar and fesc
    def test_bursty_instant_fesc_power(
        self,
        make_fstar_with_haloes: FstarFactory,
        haloes: Haloes,
        mock_bin_sources: mock.MagicMock,
    ) -> None:
        sim = make_fstar_with_haloes(
            bursty_sfr="instant", fesc_model="power", accretion_model="exp"
        )
        with mock.patch.object(
            sim.bursty_relation, "get_bursty", return_value=self.MASK
        ) as get_bursty:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        kwargs = get_bursty.call_args.kwargs
        np.testing.assert_array_equal(kwargs["mass"], haloes[1])
        assert kwargs["z"] == REDSHIFT
        assert sim.perc_switchon == pytest.approx(100 * self.MASK.mean())

        fstar = sim.fstar_relation.get(Mhalo=haloes[1])
        fesc = sim.fesc_relation.get(Mhalo=haloes[1])
        kwargs = mock_bin_sources.call_args.kwargs
        np.testing.assert_array_equal(kwargs["srcpos_mpc"], haloes[0][self.MASK])
        np.testing.assert_allclose(
            kwargs["mstar_msun"], (fesc * fstar * haloes[1])[self.MASK]
        )

    # This combionation applies mask only to fstar
    def test_bursty_instant_fesc_constant(
        self,
        make_fstar_with_haloes: FstarFactory,
        haloes: Haloes,
        mock_bin_sources: mock.MagicMock,
    ) -> None:
        sim = make_fstar_with_haloes(
            bursty_sfr="instant", fesc_model="constant", accretion_model="exp"
        )
        with mock.patch.object(
            sim.bursty_relation, "get_bursty", return_value=self.MASK
        ):
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        fstar = sim.fstar_relation.get(Mhalo=haloes[1])
        mstar = mock_bin_sources.call_args.kwargs["mstar_msun"]
        np.testing.assert_allclose(
            mstar, (sim.fesc_relation.f0_esc * fstar * haloes[1])[self.MASK]
        )

    def test_bursty_instant_no_sources_switched_on(
        self, make_fstar_with_haloes: FstarFactory, mock_bin_sources: mock.MagicMock
    ) -> None:
        sim = make_fstar_with_haloes(bursty_sfr="instant", accretion_model="exp")
        with mock.patch.object(
            sim.bursty_relation, "get_bursty", return_value=np.zeros(NUM_HALOES, bool)
        ):
            srcpos, normflux = sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        mock_bin_sources.assert_not_called()
        assert srcpos.shape == (0, 3)
        assert srcpos.dtype == np.int32
        assert normflux.shape == (0,)
        assert normflux.dtype == np.float64
        assert sim.tot_phots == 0
        assert sim.perc_switchon == 0.0

    def test_bursty_no(
        self,
        make_fstar_with_haloes: FstarFactory,
        haloes: Haloes,
        mock_bin_sources: mock.MagicMock,
    ) -> None:
        sim = make_fstar_with_haloes(bursty_sfr="no", accretion_model="exp")
        sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        assert sim.perc_switchon == 100.0
        kwargs = mock_bin_sources.call_args.kwargs
        np.testing.assert_array_equal(kwargs["srcpos_mpc"], haloes[0])


class TestFlux:
    def test_bin_sources_call(
        self,
        make_fstar_with_haloes: FstarFactory,
        haloes: Haloes,
        mock_bin_sources: mock.MagicMock,
    ) -> None:
        sim = make_fstar_with_haloes()
        with mock.patch.object(sim, "read_haloes", return_value=haloes) as read_haloes:
            sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        read_haloes.assert_called_once_with(sim.sources_dir / "haloes.txt", BOXSIZE)
        mock_bin_sources.assert_called_once()
        kwargs = mock_bin_sources.call_args.kwargs
        fstar = sim.fstar_relation.get(Mhalo=haloes[1])
        np.testing.assert_array_equal(kwargs["srcpos_mpc"], haloes[0])
        np.testing.assert_allclose(
            kwargs["mstar_msun"], sim.fesc_relation.f0_esc * fstar * haloes[1]
        )
        assert kwargs["boxsize"] == BOXSIZE / sim.cosmology.h
        assert kwargs["meshsize"] == MESHSIZE

    def test_bin_sources_return_positions(
        self, make_fstar_with_haloes: FstarFactory, mock_bin_sources: mock.MagicMock
    ) -> None:
        sim = make_fstar_with_haloes()
        srcpos, normflux = sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        # Call bin_sources with the same arguments to get the expected value
        expected_srcpos, _ = bin_sources(**mock_bin_sources.call_args.kwargs)
        np.testing.assert_array_equal(srcpos, expected_srcpos)
        # current behaviour: (numsrc, 3), evolve3D transposes it
        assert srcpos.ndim == 2 and srcpos.shape[1] == 3
        assert normflux.shape == (srcpos.shape[0],)

    def test_tot_phots(self, make_fstar_with_haloes: FstarFactory) -> None:
        sim = make_fstar_with_haloes()
        _, normflux = sim.ionizing_flux("haloes.txt", z=REDSHIFT, dt=DT)

        assert sim.tot_phots == pytest.approx(np.sum(normflux * DT * S_STAR_REF))


# ======================================================================
# ionizing_flux: expected values with the real source models
# ======================================================================

# (fstar_model, fesc_model, accretion_model, bursty_sfr)
TEST_CASES: list[TestCase] = [
    ("fgamma", "constant", "constant", "no"),
    ("dpl", "power", "EXP", "no"),
    ("dpl", "power_obs", "EXP", "no"),
    ("dpl", "thesan", "EXP", "no"),
    ("Muv", "power", "EXP", "no"),
    ("dpl", "constant", "EXP", "instant"),
    ("dpl", "power", "EXP", "instant"),
]


@pytest.fixture(scope="module")
def expected_ionizing_flux(data_dir: Path) -> dict[str, np.ndarray]:
    with np.load(data_dir / "fstar_ionizing_fluxes.npz") as data:
        return dict(data)


def case_id(case: tuple[str, ...]) -> str:
    return "-".join(case)


@pytest.mark.parametrize("case", TEST_CASES, ids=case_id)
def test_expected_ionizing_flux(
    expected_ionizing_flux: dict[str, np.ndarray], case: TestCase
) -> None:
    result = run_test_case(case)

    for key, value in result.items():
        expected = expected_ionizing_flux[f"{case_id(case)}/{key}"]
        if key == "srcpos":
            np.testing.assert_array_equal(value, expected)
        else:
            np.testing.assert_allclose(value, expected, rtol=1e-12)


# ======================================================================
# read_haloes
# ======================================================================


class TestReadHaloes:
    def test_hdf5(self, make_fstar: FstarFactory, tmp_path: Path) -> None:
        sim = make_fstar()
        h = sim.cosmology.h
        mass = np.array([1e9, 2e10, 3e11])
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        fname = tmp_path / "haloes.hdf5"
        with h5py.File(fname, "w") as f:
            f.attrs["h"] = h
            f["mass"] = mass
            f["pos"] = pos

        srcpos, srcmass = sim.read_haloes(fname, BOXSIZE)

        np.testing.assert_allclose(srcmass, mass / h)
        np.testing.assert_allclose(srcpos, pos / h)

    def test_dat(
        self,
        make_fstar: FstarFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sim = make_fstar()
        h = sim.cosmology.h
        mass = np.array([1e9, 2e10])
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        halo_list = mock.MagicMock()
        halo_list.get.side_effect = lambda var: {"m": mass, "pos": pos}[var]
        cubep3m = mock.MagicMock(return_value=halo_list)
        monkeypatch.setattr(c2ray_fstar.t2c, "HaloCubeP3MFull", cubep3m)

        fname = tmp_path / "haloes.dat"
        srcpos, srcmass = sim.read_haloes(fname, BOXSIZE)

        cubep3m.assert_called_once_with(filename=fname, box_len=BOXSIZE)
        np.testing.assert_allclose(srcmass, mass / h)
        np.testing.assert_allclose(srcpos, pos / h)

    def test_txt_fof_shift_and_wrap(
        self, make_fstar: FstarFactory, tmp_path: Path
    ) -> None:
        sim = make_fstar()
        h = sim.cosmology.h
        # positions are centred on the origin and shifted by BOXSIZE / 2
        hl = np.array(
            [
                [1e9, 0.0, 1.0, -1.0],
                [2e10, -6.0, 2.0, 3.0],  # x -> -1 -> wrapped to BOXSIZE - 1
                [3e11, 7.0, -2.0, 4.0],  # x -> 12 -> reflected to 2 * BOXSIZE - 12
            ]
        )
        fname = tmp_path / "fof_haloes.txt"
        np.savetxt(fname, hl)

        srcpos, srcmass = sim.read_haloes(fname, BOXSIZE)

        expected = hl[:, 1:] + BOXSIZE / 2
        expected[1, 0] = BOXSIZE - 1.0
        # current behaviour: values above BOXSIZE are reflected, not wrapped
        expected[2, 0] = 2 * BOXSIZE - 12.0
        np.testing.assert_allclose(srcmass, hl[:, 0] / h)
        np.testing.assert_allclose(srcpos, expected / h)

    def test_txt_halo_wrap(self, make_fstar: FstarFactory, tmp_path: Path) -> None:
        sim = make_fstar()
        h = sim.cosmology.h
        hl = np.array(
            [
                [1e9, 1.0, 2.0, 3.0],
                [2e10, -1.0, 2.0, 3.0],
                [3e11, 12.0, 4.0, 5.0],
            ]
        )
        fname = tmp_path / "halo_list.txt"
        np.savetxt(fname, hl)

        srcpos, srcmass = sim.read_haloes(fname, BOXSIZE)

        expected = hl[:, 1:].copy()
        expected[1, 0] = BOXSIZE - 1.0
        expected[2, 0] = 12.0 - BOXSIZE
        np.testing.assert_allclose(srcmass, hl[:, 0] / h)
        np.testing.assert_allclose(srcpos, expected / h)


# ======================================================================
# read_density
# ======================================================================


def expected_ndens(sim: C2Ray_fstar, overd: FloatArray) -> FloatArray:
    ndens = (
        sim.cosmology.critical_density0.cgs.value
        * sim.cosmology.Ob0
        * (1.0 + overd)
        / (sim.parameters.cosmology.mean_molecular * c.m_p)
        * (1 + REDSHIFT) ** 3
    )
    return np.maximum(ndens, 5e-6)


class TestReadDensity:
    def test_npy(self, make_fstar: FstarFactory, tmp_path: Path) -> None:
        sim = make_fstar()
        rho = np.random.default_rng(0).uniform(0.5, 2.0, size=sim.shape)
        np.save(tmp_path / "dens.npy", rho)

        sim.read_density("dens.npy", z=REDSHIFT)

        np.testing.assert_allclose(sim.ndens, expected_ndens(sim, rho - 1.0))

    def test_pkdgrav(
        self, make_fstar: FstarFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sim = make_fstar()
        overd = np.random.default_rng(0).uniform(-0.5, 1.0, size=sim.shape)
        reader = mock.MagicMock()
        reader.return_value.load_density_field.return_value = overd
        monkeypatch.setattr(c2ray_fstar.t2c, "Pkdgrav3data", reader)

        sim.read_density("dens.00100", z=REDSHIFT)

        reader.assert_called_once_with(BOXSIZE, MESHSIZE, Omega_m=sim.cosmology.Om0)
        reader.return_value.load_density_field.assert_called_once_with(
            str(sim.density_dir / "dens.00100")
        )
        np.testing.assert_allclose(sim.ndens, expected_ndens(sim, overd))

    def test_density_floor(self, make_fstar: FstarFactory, tmp_path: Path) -> None:
        sim = make_fstar()
        rho = np.ones(sim.shape)
        rho[0, 0, 0] = 0.0
        np.save(tmp_path / "dens.npy", rho)

        sim.read_density("dens.npy", z=REDSHIFT)

        assert sim.ndens[0, 0, 0] == 5e-6
        assert sim.ndens.min() == 5e-6
