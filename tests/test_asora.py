import itertools
import math
from collections import Counter

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

    xi = abs(di / dk)
    xj = abs(dj / dk)
    path = math.sqrt(1 + xi**2 + xj**2)
    return xi, xj, path


def expected_offsets(R: float, N: int, q_max: int) -> list[tuple[int, int, int]]:
    """Cells kept in the LUT: inside the sphere and, if periodic, inside the box."""
    assert libasora is not None
    radius2 = int(R * R)
    periodic = libasora.is_periodic_mode_active()

    kept = []
    for q in range(q_max + 1):
        for s in range(libasora.cells_in_shell(q)):
            pos = libasora.shell2cart(q, s)
            if sum(x * x for x in pos) > radius2:
                continue
            if periodic and not libasora.in_domain((0, 0, 0), pos, N):
                continue
            kept.append(pos)
    return kept


def shell_starts(lut) -> list[int]:
    """First LUT index of each shell, plus the total as sentinel."""
    counts = Counter(abs(item.di) + abs(item.dj) + abs(item.dk) for item in lut)
    return [0, *itertools.accumulate(counts[q] for q in range(max(counts) + 1))]


# (R, N): sphere-limited, fractional radius, box-limited (even and odd), R < 1.
LUT_CASES = [(10.0, 64), (7.5, 64), (30.0, 32), (30.0, 31), (0.5, 16)]


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
            libasora.pack_offset(0, 1, -512)
        with pytest.raises(RuntimeError):
            libasora.pack_offset(1, -512, 0)
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

    def test_shortchar_lut_invalid(self, init_device) -> None:
        assert libasora is not None
        with pytest.raises(ValueError):
            libasora.create_shortchar_lut(10.0, 0)

        # q_max = ceil(sqrt(3) * 400) > 512 cannot be packed.
        with pytest.raises(RuntimeError):
            libasora.create_shortchar_lut(400.0, 1024)

    @pytest.mark.parametrize("R, N", LUT_CASES)
    def test_shortchar_lut_created(self, init_device, R: float, N: int) -> None:
        assert libasora is not None

        q_max = libasora.create_shortchar_lut(R, N)
        assert q_max == math.ceil(min(math.sqrt(3) * R, 1.5 * N))

        lut = libasora.get_shortchar_lut()
        offsets = [(item.di, item.dj, item.dk) for item in lut]
        assert offsets == expected_offsets(R, N, q_max)

    @pytest.mark.parametrize("R, N", LUT_CASES)
    def test_shortchar_lut_check_items(self, init_device, R: float, N: int) -> None:
        assert libasora is not None

        libasora.create_shortchar_lut(R, N)
        lut = libasora.get_shortchar_lut()
        start = shell_starts(lut)
        bank_size = max(b - a for a, b in itertools.pairwise(start))

        def corner(q: int, index: int) -> np.ndarray:
            """Cell held in column-density slot index, as seen from a cell of shell q."""
            n, s = divmod(index, bank_size)
            other_q = q - (q - n) % 4
            assert other_q < q
            assert s < start[other_q + 1] - start[other_q]
            other = lut[start[other_q] + s]
            return np.array([other.di, other.dj, other.dk])

        # The source cell (q = 0) is raytraced separately: its entry is unused.
        for item in lut[1:]:
            pos = (item.di, item.dj, item.dk)
            q = sum(abs(x) for x in pos)

            # Check that the path and geometric factors match.
            dx, dy, path = geometric_factors(*pos)
            assert item.dx == pytest.approx(dx)
            assert item.dy == pytest.approx(dy)
            assert item.path == pytest.approx(path)

            # The ray from the source crosses the previous plane along the major axis
            # at pos * (1 - 1 / |major|): the weighted corners must land on it.
            weights = [dx * dy, (1 - dx) * dy, (1 - dy) * dx, (1 - dx) * (1 - dy)]
            crossing = sum(
                ws * corner(q, index)
                for ws, index in zip(weights, item.indices)
                if ws > 0
            )
            expected = np.array(pos) * (1 - 1 / max(abs(x) for x in pos))
            assert crossing == pytest.approx(expected)

    @pytest.mark.parametrize("R, N", LUT_CASES)
    def test_shortchar_lut_check_order(self, init_device, R: float, N: int) -> None:
        assert libasora is not None

        libasora.create_shortchar_lut(R, N)
        lut = libasora.get_shortchar_lut()
        start = shell_starts(lut)

        # Entries are sorted in lexicographic order of (di, dj, dk) in each q-shell.
        for a, b in itertools.pairwise(start):
            ijk = np.array(
                [(item.di, item.dj, item.dk) for item in lut[a:b]], dtype=np.int32
            ).reshape(-1, 3)
            indices = np.lexsort(ijk.T[::-1])
            assert (ijk == ijk[indices]).all()


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

    @pytest.mark.parametrize("q", range(Q_MAX))
    def test_index2shell(self, q: int) -> None:
        """Every cumulative index is split back into the shell that holds it."""
        assert libasora is not None

        base = libasora.cells_to_shell(q - 1)
        for s in range(libasora.cells_in_shell(q)):
            assert libasora.index2shell(base + s) == (q, s)

    def test_index2shell_shell_edges(self) -> None:
        """The first and last index of a shell are the ones a cube-root estimate is
        most likely to misplace, so pin them down across the whole range."""
        assert libasora is not None

        assert libasora.index2shell(0) == (0, 0)
        for q in range(Q_MAX):
            first = libasora.cells_to_shell(q - 1)
            last = libasora.cells_to_shell(q) - 1

            assert libasora.index2shell(first) == (q, 0)
            assert libasora.index2shell(last) == (q, libasora.cells_in_shell(q) - 1)

    def test_index2shell_rejects_negative(self) -> None:
        assert libasora is not None

        with pytest.raises(ValueError):
            libasora.index2shell(-1)

        with pytest.raises(TypeError):
            libasora.index2shell(1.5)


class TestModuleInterface:
    assert libasora is not None

    def test_no_device_init(self):
        assert not libasora.is_device_init()

    def test_device_init(self, init_device):
        assert libasora.is_device_init()

    @staticmethod
    def create_density_data(size: int) -> np.ndarray:
        dens = np.full(size**3, 0.5, dtype=np.float64)
        return dens

    def test_density_to_device_no_init(self):
        with pytest.raises(RuntimeError):
            libasora.density_to_device(self.create_density_data(16))

    def test_density_to_device(self, init_device):
        # One argument required
        with pytest.raises(TypeError):
            libasora.density_to_device()

        # np.float64 array required
        with pytest.raises(TypeError):
            libasora.density_to_device(np.ones(10, dtype=np.int32))

        assert libasora is not None
        libasora.density_to_device(self.create_density_data(16))
        libasora.density_to_device(self.create_density_data(64))
        libasora.density_to_device(self.create_density_data(32))

    @staticmethod
    def create_photo_table_data(num_tau: int) -> tuple[np.ndarray, np.ndarray]:
        thin = np.linspace(-20, 4, num_tau + 1, dtype=np.float64)
        thick = np.linspace(-20, 4, num_tau + 1, dtype=np.float64)
        return thin, thick

    def test_photo_table_to_device_no_init(self):
        with pytest.raises(RuntimeError):
            libasora.photo_table_to_device(*self.create_photo_table_data(80))

    def test_photo_table_to_device(self, init_device):
        # Two arguments required
        with pytest.raises(TypeError):
            libasora.photo_table_to_device(np.array([]))

        # Both arguments must be np.float64 arrays
        with pytest.raises(TypeError):
            libasora.photo_table_to_device(
                np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float64)
            )

        assert libasora is not None
        libasora.photo_table_to_device(*self.create_photo_table_data(80))
        libasora.photo_table_to_device(*self.create_photo_table_data(100))
        libasora.photo_table_to_device(*self.create_photo_table_data(90))

    @staticmethod
    def create_source_data(num_sources: int) -> tuple[np.ndarray, np.ndarray]:
        src_pos = np.arange(0, 3 * num_sources, dtype=np.int32)
        norm_flux = np.ones(num_sources, dtype=np.float64)
        return src_pos, norm_flux

    def test_source_data_to_device_no_init(self):
        with pytest.raises(RuntimeError):
            libasora.source_data_to_device(*self.create_source_data(50))

    def test_source_data_to_device(self, init_device):
        # Two arguments required
        with pytest.raises(TypeError):
            libasora.source_data_to_device(np.array([]))

        # First argument is array np.int32, second argument is array np.float64
        with pytest.raises(TypeError):
            libasora.source_data_to_device(
                np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64)
            )

        assert libasora is not None
        libasora.source_data_to_device(*self.create_source_data(50))
        libasora.source_data_to_device(*self.create_source_data(100))
        libasora.source_data_to_device(*self.create_source_data(80))

    @staticmethod
    def create_timestep_data(
        mesh_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = mesh_size**3
        xh = np.full(size, 1e-4, dtype=np.float64)
        temp = np.full(size, 1e4, dtype=np.float64)
        clump = np.ones(size, dtype=np.float64)
        return xh, temp, clump

    def test_timestep_data_to_device_no_init(self):
        with pytest.raises(RuntimeError):
            libasora.timestep_data_to_device(*self.create_timestep_data(8))

    def test_timestep_data_to_device(self, init_device):
        # Three arguments required
        with pytest.raises(TypeError):
            libasora.timestep_data_to_device(np.zeros(8), np.zeros(8))

        # All three must be np.float64 arrays
        with pytest.raises(TypeError):
            libasora.timestep_data_to_device(
                np.ones(8, dtype=np.float32), np.ones(8), np.ones(8)
            )
        with pytest.raises(TypeError):
            libasora.timestep_data_to_device(np.ones(8), [0.0] * 8, np.ones(8))

        assert libasora is not None
        libasora.timestep_data_to_device(*self.create_timestep_data(16))
        libasora.timestep_data_to_device(*self.create_timestep_data(64))
        libasora.timestep_data_to_device(*self.create_timestep_data(32))

    def test_average_fraction_to_device_no_init(self):
        with pytest.raises(RuntimeError):
            libasora.average_fraction_to_device(self.create_density_data(8))

    def test_average_fraction_to_device(self, init_device):
        # One argument required
        with pytest.raises(TypeError):
            libasora.average_fraction_to_device()

        # np.float64 array required
        with pytest.raises(TypeError):
            libasora.average_fraction_to_device(np.ones(8, dtype=np.float32))

        assert libasora is not None
        libasora.average_fraction_to_device(self.create_density_data(16))
        libasora.average_fraction_to_device(self.create_density_data(64))
        libasora.average_fraction_to_device(self.create_density_data(32))

    def test_average_fraction_to_host_no_init(self):
        with pytest.raises(RuntimeError):
            libasora.average_fraction_to_host(self.create_density_data(8))

    def test_average_fraction_to_host(self, init_device):
        # One argument required
        with pytest.raises(TypeError):
            libasora.average_fraction_to_host()

        # np.float64 array required
        with pytest.raises(TypeError):
            libasora.average_fraction_to_host(np.ones(8, dtype=np.float32))

        # Keep integration-level sanity check with a matching density upload.
        dens = np.full(20**3, 0.5, dtype=np.float64)
        libasora.density_to_device(dens)

    def test_is_periodic_mode_active(self):
        assert isinstance(libasora.is_periodic_mode_active(), bool)

    @pytest.mark.skipif(
        not libasora.is_periodic_mode_active(),
        reason="libasora not compiled with periodic boundary conditions",
    )
    @pytest.mark.parametrize("N", [10, 11], ids=["even", "odd"])
    def test_in_domain_periodic(self, N: int):
        # Unused in periodic mode
        pos = 0, 0, 0

        N = 10
        q = N // 2 + 2
        low_lim = -(N // 2)
        up_lim = N // 2 + N % 1 - 1

        assert libasora is not None
        for off in itertools.product(range(-q, q + 1), repeat=3):
            assert libasora.in_domain(pos, off, N) == all(
                low_lim <= x <= up_lim for x in off
            )

    @pytest.mark.skipif(
        libasora.is_periodic_mode_active(),
        reason="libasora compiled with periodic boundary conditions",
    )
    def test_in_domain_non_periodic(self):
        N = 10
        for pos in itertools.product(range(N // 2, N), repeat=3):
            for off in itertools.product(range(-3, 4), repeat=3):
                assert libasora.in_domain(pos, off, N) == all(
                    0 <= x + y < N for x, y in zip(pos, off)
                )

    def test_average_fraction_to_host_unallocated(self, init_device):
        """Nothing has pushed the average fraction yet, so there is nothing to read."""
        assert libasora is not None
        with pytest.raises(RuntimeError):
            libasora.average_fraction_to_host(np.zeros(8, dtype=np.float64))

    def test_average_fraction_round_trip(self, init_device):
        """The path of a rank that receives the field by broadcast.

        It pushes the values it was given, overwriting whatever the last chemistry
        pass left on the device, and must read exactly those back.
        """
        rng = np.random.default_rng(2024)
        xh = rng.uniform(0.0, 1.0, size=512)
        ones = np.ones_like(xh)

        assert libasora is not None
        libasora.timestep_data_to_device(xh, 1e4 * ones, ones)

        xh_av = rng.uniform(0.0, 1.0, size=512)
        libasora.average_fraction_to_device(xh_av)

        received = np.zeros_like(xh_av)
        libasora.average_fraction_to_host(received)

        assert np.array_equal(received, xh_av)
        assert not np.array_equal(received, xh)

    def test_average_fraction_to_device_allocates(self, init_device):
        """to_device is enough on its own: it grows the array it writes into."""
        assert libasora is not None
        xh_av = np.linspace(0.0, 1.0, 64, dtype=np.float64)
        libasora.average_fraction_to_device(xh_av)

        received = np.zeros_like(xh_av)
        libasora.average_fraction_to_host(received)

        assert np.array_equal(received, xh_av)
