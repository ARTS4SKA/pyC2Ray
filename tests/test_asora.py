import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pyc2ray.lib import libasoratest
from pyc2ray.load_extensions import libasora

Q_MAX = 100


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

    @pytest.mark.parametrize("q", range(Q_MAX))
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

    def test_photo_table_to_device(self, init_device):
        # Two arguments required
        with pytest.raises(TypeError):
            libasora.photo_table_to_device(np.array([]))

        # Both arguments must be np.float64 arrays
        with pytest.raises(TypeError):
            libasora.photo_table_to_device(
                np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float64)
            )

        def create_photo_table_data(num_tau: int) -> tuple[np.ndarray, np.ndarray]:
            thin = np.linspace(-20, 4, num_tau + 1, dtype=np.float64)
            thick = np.linspace(-20, 4, num_tau + 1, dtype=np.float64)
            return thin, thick

        assert libasora is not None
        libasora.photo_table_to_device(*create_photo_table_data(80))
        libasora.photo_table_to_device(*create_photo_table_data(100))
        libasora.photo_table_to_device(*create_photo_table_data(90))

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

    def test_prepare_grid_buffers(self, init_device):
        # One argument required
        with pytest.raises(TypeError):
            libasora.prepare_grid_buffers()

        # More than two arguments is invalid
        with pytest.raises(TypeError):
            libasora.prepare_grid_buffers(16, False, 0)

        # Exercise both default mode and exact-size-forcing mode.
        libasora.prepare_grid_buffers(16)
        libasora.prepare_grid_buffers(16, True)
        libasora.prepare_grid_buffers(24, False)

        # Verify subsequent number-density upload remains functional.
        dens = np.full(24**3, 0.5, dtype=np.float64)
        libasora.density_to_device(dens)

    def test_prepare_grid_buffers_forced_mode_idempotent(self, init_device):
        # Repeated exact-size enforcement should be idempotent and not raise.
        libasora.prepare_grid_buffers(20, True)
        libasora.prepare_grid_buffers(20, True)
        libasora.prepare_grid_buffers(20, True)

        # Keep integration-level sanity check with a matching density upload.
        dens = np.full(20**3, 0.5, dtype=np.float64)
        libasora.density_to_device(dens)

    def test_pack_unpack_offsets(self) -> None:
        assert libasora is not None
        for pos in itertools.product(range(-Q_MAX, Q_MAX + 1), repeat=3):
            packed = libasora.pack_offset(*pos)
            unpacked = libasora.unpack_offset(packed)
            assert pos == unpacked

    def test_cells_in_shell(self) -> None:
        assert libasora is not None
        assert libasora.cells_in_shell(0) == 1
        for q in range(1, Q_MAX):
            assert libasora.cells_in_shell(q) == 4 * q**2 + 2

    def test_cells_to_shell(self) -> None:
        q_tot = 1
        assert libasora is not None
        assert libasora.cells_to_shell(0) == q_tot
        for q in range(1, Q_MAX):
            q_tot += 4 * q**2 + 2
            assert libasora.cells_to_shell(q) == q_tot

    @pytest.mark.parametrize(
        "pos",
        [
            (-512, -512, -512),
            (511, 511, 511),
            (512, 0, 0),
            (0, 512, 0),
            (0, 0, 512),
        ],
    )
    def test_pack_unpack_offsets_edge_cases(self, pos: tuple[int, int, int]) -> None:
        assert libasora is not None
        packed = libasora.pack_offset(*pos)
        unpacked = libasora.unpack_offset(packed)
        assert pos == unpacked

    def test_raytracing_lut(self, init_device) -> None:
        def path(di: int, dj: int, dk: int) -> float:
            if di == 0 and dj == 0 and dk == 0:
                return 0.5

            di2 = di * di
            dj2 = dj * dj
            dk2 = dk * dk
            delta_max = np.maximum(di2, np.maximum(dj2, dk2))
            return np.sqrt((di2 + dj2 + dk2) / delta_max)

        def geometric_factors(di: int, dj: int, dk: int) -> tuple[float, float]:
            if di == 0 and dj == 0 and dk == 0:
                return 0.0, 0.0
            ai = abs(di)
            aj = abs(dj)
            ak = abs(dk)

            if ak >= ai and ak >= aj:
                pass
            elif aj >= ai and aj >= ak:
                dj, dk = dk, dj
            else:  # ai >= aj and ai >= ak
                di, dk = dk, di
                di, dj = dj, di

            dx = abs(math.copysign(1, di) - di / abs(dk))
            dy = abs(math.copysign(1, dj) - dj / abs(dk))
            return dx, dy

        q_max = 50
        assert libasora is not None
        lut = libasora.get_raytracing_lut(q_max)

        assert len(lut) == libasora.cells_to_shell(q_max)

        for item in lut:
            # Check that the path and geometric factors match.
            assert item.path == pytest.approx(path(item.di, item.dj, item.dk))

            dx, dy = geometric_factors(item.di, item.dj, item.dk)
            assert item.dx == pytest.approx(dx)
            assert item.dy == pytest.approx(dy)

            weights = [(1 - dx) * (1 - dy), (1 - dy) * dx, (1 - dx) * dy, dx * dy]

            # Check that the interpolation indices are correct.
            for ws, index in zip(weights, item.indices):
                if ws > 0:
                    other_item = lut[index]
                    assert abs(item.di - other_item.di) <= 1
                    assert abs(item.dj - other_item.dj) <= 1
                    assert abs(item.dk - other_item.dk) <= 1

        # Entries are sorted in lexicographic order of (di, dj, dk)
        # in each q-shell.
        start = 0
        for q in range(q_max + 1):
            ncells = libasora.cells_in_shell(q)
            s = slice(start, start + ncells)
            ijk = np.array(
                [(item.di, item.dj, item.dk) for item in lut[s]], dtype=np.int32
            )

            # Assert lexicographic order.
            indices = np.lexsort(ijk.T[::-1])
            assert (ijk == ijk[indices]).all()

            start += ncells

    @pytest.mark.parametrize("q_max", [50, 100, 150, 200, 250])
    def test_benchmark_create_raytracing_lut(
        self, benchmark, init_device, q_max: int
    ) -> None:
        assert libasora is not None
        libasora.create_raytracing_lut(q_max)
        benchmark(libasora.create_raytracing_lut, q_max)
