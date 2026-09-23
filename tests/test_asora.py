import itertools
import math

import numpy as np
import pytest

from pyc2ray.load_extensions import libasora

Q_MAX = 100

if libasora is None:
    pytest.skip("libasora.so missing, skipping tests", allow_module_level=True)


def geometric_factors(di: int, dj: int, dk: int) -> tuple[float, float, float]:
    if di == 0 and dj == 0 and dk == 0:
        return 0.0, 0.0, 0.5

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

    xi = di / abs(dk)
    xj = dj / abs(dk)

    dx = 1 - abs(xi)
    dy = 1 - abs(xj)
    path = math.sqrt(1 + xi**2 + xj**2)
    return dx, dy, path


class TestShortCharacteristicsInterpolation:
    @pytest.mark.parametrize("dk", range(1, 11))
    def test_compute_shortchar_factors(self, dk: int) -> None:
        assert libasora is not None
        ak = abs(dk)
        for di, dj in itertools.product(range(-ak, ak + 1), repeat=2):
            dx, dy, path = libasora.compute_shortchar_factors(di, dj, dk)
            exp_dx, exp_dy, exp_path = geometric_factors(di, dj, dk)
            assert dx == pytest.approx(exp_dx)
            assert dy == pytest.approx(exp_dy)
            assert path == pytest.approx(exp_path)

    def test_pack_offsets_out_of_bounds(self) -> None:
        assert libasora is not None
        with pytest.raises(RuntimeError):
            libasora.pack_offset(512, 1, -1)
        with pytest.raises(RuntimeError):
            libasora.pack_offset(0, 513, 0)
        with pytest.raises(RuntimeError):
            libasora.pack_offset(0, 0, -513)

    def test_pack_unpack_offsets(self) -> None:
        assert libasora is not None
        for pos in itertools.product(range(-Q_MAX, Q_MAX + 1), repeat=3):
            packed = libasora.pack_offset(*pos)
            unpacked = libasora.unpack_offset(packed)
            assert pos == unpacked

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

    def test_shortchar_lut_not_created(self, init_device) -> None:
        assert libasora is not None
        with pytest.raises(RuntimeError):
            libasora.get_shortchar_lut()

    def test_shortchar_lut_created(self, init_device) -> None:
        assert libasora is not None

        libasora.create_shortchar_lut(10)
        lut = libasora.get_shortchar_lut()
        assert len(lut) == libasora.cells_to_shell(10)

    def test_shortchar_lut_check_items(self, init_device) -> None:
        assert libasora is not None

        libasora.create_shortchar_lut(50)
        lut = libasora.get_shortchar_lut()

        for item in lut:
            # Check that the path and geometric factors match.
            dx, dy, path = geometric_factors(item.di, item.dj, item.dk)
            assert item.dx == pytest.approx(dx)
            assert item.dy == pytest.approx(dy)
            assert item.path == pytest.approx(path)

            weights = [(1 - dx) * (1 - dy), (1 - dy) * dx, (1 - dx) * dy, dx * dy]

            # Check that the interpolation indices are correct.
            for ws, index in zip(weights, item.indices):
                if ws > 0:
                    other_item = lut[index]
                    assert abs(item.di - other_item.di) <= 1
                    assert abs(item.dj - other_item.dj) <= 1
                    assert abs(item.dk - other_item.dk) <= 1

    def test_shortchar_lut_check_order(self, init_device) -> None:
        assert libasora is not None

        libasora.create_shortchar_lut(50)
        lut = libasora.get_shortchar_lut()

        # Entries are sorted in lexicographic order of (di, dj, dk)
        # in each q-shell.
        start = 0
        for q in range(51):
            ncells = libasora.cells_in_shell(q)
            s = slice(start, start + ncells)
            ijk = np.array(
                [(item.di, item.dj, item.dk) for item in lut[s]], dtype=np.int32
            )

            # Assert lexicographic order.
            indices = np.lexsort(ijk.T[::-1])
            assert (ijk == ijk[indices]).all()

            start += ncells


class TestOctahedron:
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

    @pytest.mark.parametrize("q", range(Q_MAX))
    def test_shell_mapping(self, q: int) -> None:
        assert libasora is not None

        cells: set[tuple[int, int, int]] = set()
        q_max = 4 * q**2 + 2 if q > 0 else 1
        for s in range(q_max):
            # Check value makes sense
            ijk = libasora.shell2cart(q, s)
            assert q == sum(abs(x) for x in ijk)

            # Check it's unique
            assert ijk not in cells
            cells.add(ijk)

            # Check inverse function
            assert (q, s) == libasora.cart2shell(*ijk)


class TestModuleInterface:
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
