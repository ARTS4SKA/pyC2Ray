from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pyc2ray.load_extensions import libasora_He as libasora

try:
    import pyc2ray.lib.libasoratest as libasoratest
except ImportError:
    libasoratest = None


@pytest.mark.skipif(libasora is None, reason="libasora.so missing, skipping tests")
class TestLibasoraTest:
    def test_path_in_cell(self) -> None:
        def create_path_in_cell_data(N: int) -> NDArray:
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
        path = libasoratest.path_in_cell((N, N, N))
        expected = create_path_in_cell_data(N)

        assert np.allclose(path, expected)

    def test_geometric_factors(self) -> None:
        def create_geometric_factors_data(N: int) -> NDArray:
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
        facts = libasoratest.geometric_factors((N, N, N))
        expected = create_geometric_factors_data(N)

        assert np.allclose(facts, expected)

    def test_cell_interpolator(self, data_dir: Path) -> None:
        rng = np.random.default_rng(seed=42)
        N = 11
        dens = rng.random((N, N, N), dtype=np.float64)

        cdens = libasoratest.cell_interpolator(dens)
        expected_output = np.load(data_dir / "cell_interpolator_output.npy")

        assert np.allclose(cdens, expected_output)

    Q_MAX = 100

    def test_cells_in_shell(self) -> None:
        assert libasoratest.cells_in_shell(0) == 1
        for q in range(1, self.Q_MAX):
            assert libasoratest.cells_in_shell(q) == 4 * q**2 + 2

    def test_cells_to_shell(self) -> None:
        q_tot = 1
        assert libasoratest.cells_to_shell(0) == q_tot
        for q in range(1, self.Q_MAX):
            q_tot += 4 * q**2 + 2
            assert libasoratest.cells_to_shell(q) == q_tot

    @pytest.mark.parametrize("q", range(0, Q_MAX))
    def test_shell_mapping(self, q: int) -> None:
        cells: set[tuple[int, int, int]] = set()
        q_max = 4 * q**2 + 2 if q > 0 else 1
        for s in range(q_max):
            # Check value makes sense
            ijk = libasoratest.linthrd2cart(q, s)
            assert q == sum(abs(x) for x in ijk)

            # Check it's unique
            assert ijk not in cells
            cells.add(ijk)

            # Check inverse function
            assert (q, s) == libasoratest.cart2linthrd(*ijk)


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
