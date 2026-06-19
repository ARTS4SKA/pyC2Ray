from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import pyc2ray.constants as c
from pyc2ray.load_extensions import libasora_He as libasora
from pyc2ray.radiation.blackbody import BlackBodySource, make_tau_table

libasoratest: ModuleType | None

try:
    from pyc2ray.lib import libasoratest  # type: ignore
except ImportError:
    libasoratest = None

LogtauSpace = tuple[float, float, int]


@pytest.mark.skipif(libasoratest is None, reason="libasora.so missing, skipping tests")
class TestLibasoraTest:
    def test_path_in_cell(self) -> None:
        def create_path_in_cell_data(N: int) -> np.ndarray:
            """Return the length of the ray intersecting cell at pos emitted from pos0"""
            N2 = N // 2
            di, dj, dk = np.mgrid[-N2 : N2 + 1, -N2 : N2 + 1, -N2 : N2 + 1]

            di2 = di * di
            dj2 = dj * dj
            dk2 = dk * dk
            delta_max = np.maximum(di2, np.maximum(dj2, dk2))

            paths = np.sqrt((di2 + dj2 + dk2) / delta_max)
            paths[N2, N2, N2] = 0.5
            return paths

        N = 11
        assert libasoratest is not None
        path = libasoratest.path_in_cell((N, N, N))
        expected = create_path_in_cell_data(N)

        assert np.allclose(path, expected)

    def test_geometric_factors(self) -> None:
        def create_geometric_factors_data(N: int) -> np.ndarray:
            """Return the geometric interpolation factors (weights) for the 4 adjacent cells"""
            N2 = N // 2
            grid = np.mgrid[-N2 : N2 + 1, -N2 : N2 + 1, -N2 : N2 + 1]
            indices = np.abs(grid).argsort(axis=0)
            di, dj, dk = np.take_along_axis(grid, indices, axis=0)

            dx = np.abs(np.copysign(1, di) - di / np.abs(dk))
            dy = np.abs(np.copysign(1, dj) - dj / np.abs(dk))

            w1 = (1 - dx) * (1 - dy)
            w2 = (1 - dy) * dx
            w3 = (1 - dx) * dy
            w4 = dx * dy

            facts = np.stack((w1, w2, w3, w4), axis=-1)
            facts[dk == 0] = 0.0
            return facts

        N = 11
        assert libasoratest is not None
        facts = libasoratest.geometric_factors((N, N, N))
        expected = create_geometric_factors_data(N)

        assert np.allclose(facts, expected)

    def test_cell_interpolator(self, data_dir: Path) -> None:
        rng = np.random.default_rng(seed=42)
        N = 11
        dens = rng.random((N, N, N), dtype=np.float64)

        assert libasoratest is not None
        cdens = libasoratest.cell_interpolator(dens)
        expected_output = np.load(data_dir / "cell_interpolator_output.npy")

        assert np.allclose(cdens, expected_output)

    Q_MAX = 100

    def test_cells_in_shell(self) -> None:
        assert libasoratest is not None
        assert libasoratest.cells_in_shell(0) == 1
        for q in range(1, self.Q_MAX):
            assert libasoratest.cells_in_shell(q) == 4 * q**2 + 2

    def test_cells_to_shell(self) -> None:
        q_tot = 1
        assert libasoratest is not None
        assert libasoratest.cells_to_shell(0) == q_tot
        for q in range(1, self.Q_MAX):
            q_tot += 4 * q**2 + 2
            assert libasoratest.cells_to_shell(q) == q_tot

    @pytest.mark.parametrize("q", range(Q_MAX))
    def test_shell_mapping(self, q: int) -> None:
        cells: set[tuple[int, int, int]] = set()
        q_max = 4 * q**2 + 2 if q > 0 else 1
        for s in range(q_max):
            # Check value makes sense
            assert libasoratest is not None
            ijk = libasoratest.linthrd2cart(q, s)
            assert q == sum(abs(x) for x in ijk)

            # Check it's unique
            assert ijk not in cells
            cells.add(ijk)

            # Check inverse function
            assert (q, s) == libasoratest.cart2linthrd(*ijk)

    @pytest.fixture(scope="class")
    @classmethod
    def logtau_space(cls) -> LogtauSpace:
        return -20.0, 24.0 / 2000, 2000

    @pytest.fixture(scope="class")
    @classmethod
    def taus(cls, logtau_space: LogtauSpace) -> np.ndarray:
        minltau, dlogtau, ntau = logtau_space
        maxltau = minltau + dlogtau * ntau
        return make_tau_table(minltau, maxltau, ntau)[0]

    @pytest.fixture(scope="class")
    @classmethod
    def tables(cls, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ion_freq_HI = c.ev2hz * 13.598
        rad = BlackBodySource(5e4, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8)

        return rad.make_photo_tables(taus)

    TEST_TAUS = (0.9e-20, 1e-20, 1.1e-20, 0.123, 1.0, 123, 0.999e4, 1e4, 1.1e4)

    @pytest.mark.parametrize("tau", TEST_TAUS)
    def test_log_table_index(
        self, taus: np.ndarray, logtau_space: LogtauSpace, tau: float
    ) -> None:
        minlogtau, dlogtau, ntau = logtau_space
        maxlogtau = minlogtau + dlogtau * ntau

        assert libasoratest is not None
        i0, i1, res = libasoratest.log_table_index(tau, logtau_space)
        exp_i0 = min(ntau, max(0, np.searchsorted(taus, tau, side="right").item() - 1))
        assert i0 == exp_i0
        assert i1 == min(ntau, max(0, exp_i0 + 1))

        ltau = (i0 + res) * dlogtau + minlogtau
        exp_ltau = min(maxlogtau, max(minlogtau, np.log10(tau)))
        assert ltau == pytest.approx(exp_ltau)

    @pytest.mark.parametrize("tau", TEST_TAUS)
    def test_photo_table_lookup_thin(
        self,
        taus: np.ndarray,
        logtau_space: LogtauSpace,
        tables: tuple[np.ndarray, np.ndarray],
        tau: float,
    ) -> None:
        tau_in = tau
        tau_out = tau_in + 1e-8
        assert libasoratest is not None
        res = libasoratest.photo_table_lookup(tau_in, tau_out, *tables, logtau_space)

        # Interpolate thin table in log space with numpy
        rate = np.interp(np.log10(tau_out), np.log10(taus), tables[0])
        exp_res = (tau_out - tau_in) * rate

        assert res == pytest.approx(exp_res)

    @pytest.mark.parametrize("tau", TEST_TAUS)
    def test_photo_table_lookup_thick(
        self,
        taus: np.ndarray,
        logtau_space: LogtauSpace,
        tables: tuple[np.ndarray, np.ndarray],
        tau: float,
    ) -> None:
        tau_in = tau
        tau_out = tau_in + 1.0
        assert libasoratest is not None
        res = libasoratest.photo_table_lookup(tau_in, tau_out, *tables, logtau_space)

        # Interpolate thick table in log space with numpy
        rates = np.interp(np.log10([tau_in, tau_out]), np.log10(taus), tables[1])
        exp_res = rates[0] - rates[1]

        assert res == pytest.approx(exp_res)


@pytest.mark.skipif(libasora is None, reason="libasora.so missing, skipping tests")
class TestLibasora:
    def test_device_init(self, init_device):
        libasora.is_device_init()

    def test_density_to_device(self, init_device):
        # One argument required
        with pytest.raises(TypeError):
            libasora.density_to_device()

        # np.float64 array required
        with pytest.raises(TypeError):
            libasora.density_to_device(np.ones(10, dtype=np.int32))

        def create_density_data(mesh_size: int) -> np.ndarray:
            dens = np.full(mesh_size**3, 0.5, dtype=np.float64)
            return dens

        assert libasora is not None
        libasora.density_to_device(create_density_data(16))
        libasora.density_to_device(create_density_data(64))
        libasora.density_to_device(create_density_data(32))

    def test_photo_tables_to_device(self, init_device):
        # Two arguments required
        with pytest.raises(TypeError):
            libasora.photo_tables_to_device(np.array([]))

        # Both arguments must be np.float64 arrays
        with pytest.raises(TypeError):
            libasora.photo_tables_to_device(
                np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float64)
            )

        def create_photo_table_data(
            num_tau: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            thin = np.linspace(-20, 4, num_tau + 1, dtype=np.float64)
            thick = np.linspace(-20, 4, num_tau + 1, dtype=np.float64)
            return thin, thick, thin, thick

        assert libasora is not None
        libasora.photo_tables_to_device(*create_photo_table_data(80))
        libasora.photo_tables_to_device(*create_photo_table_data(100))
        libasora.photo_tables_to_device(*create_photo_table_data(90))

    def test_cooling_tables_to_device(self, init_device):
        # Five arguments required
        with pytest.raises(TypeError):
            libasora.cooling_tables_to_device((np.array([]),) * 4)

        # Arguments must be np.float64 arrays
        with pytest.raises(TypeError):
            libasora.cooling_tables_to_device(
                np.ones(10, dtype=np.float32), (np.zeros(10, dtype=np.float64),) * 4
            )

        def create_cooling_table_data(
            num_tau: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            cool = np.linspace(1, 9, num_tau + 1, dtype=np.float64)
            return (cool,) * 5

        assert libasora is not None
        libasora.cooling_tables_to_device(*create_cooling_table_data(80))
        libasora.cooling_tables_to_device(*create_cooling_table_data(100))
        libasora.cooling_tables_to_device(*create_cooling_table_data(90))

    def test_source_data_to_device(self, init_device):
        # Two arguments required
        with pytest.raises(TypeError):
            libasora.source_data_to_device(np.array([]))

        # First argument is array np.int32, second argument is array np.float64
        with pytest.raises(TypeError):
            libasora.source_data_to_device(
                np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64)
            )

        def create_source_data(num_sources: int) -> tuple[np.ndarray, np.ndarray]:
            src_pos = np.arange(0, 3 * num_sources, dtype=np.int32)
            norm_flux = np.ones(num_sources, dtype=np.float64)
            return src_pos, norm_flux

        assert libasora is not None
        libasora.source_data_to_device(*create_source_data(50))
        libasora.source_data_to_device(*create_source_data(100))
        libasora.source_data_to_device(*create_source_data(80))
