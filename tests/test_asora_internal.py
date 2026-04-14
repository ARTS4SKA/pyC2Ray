from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

try:
    from pyc2ray.lib import libasoratest as asoratest
except ImportError:
    pytest.skip("libasoratest.so missing, skipping tests", allow_module_level=True)


def test_path_in_cell(data_dir: Path) -> None:
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
    path = asoratest.path_in_cell((N, N, N))
    expected = create_path_in_cell_data(N)

    assert np.allclose(path, expected)


def test_geometric_factors(data_dir: Path) -> None:
    def create_geometric_factors_data(N: int) -> NDArray:
        """Return the length of the ray intersecting cell at pos emitted from pos0"""
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
    facts = asoratest.geometric_factors((N, N, N))
    expected = create_geometric_factors_data(N)

    assert np.allclose(facts, expected)


def test_cell_interpolator(data_dir: Path) -> None:
    rng = np.random.default_rng(seed=42)
    N = 11
    dens = rng.random((N, N, N), dtype=np.float64)

    cdens = asoratest.cell_interpolator(dens)
    expected_output = np.load(data_dir / "cell_interpolator_output.npy")

    assert np.allclose(cdens, expected_output)


Q_MAX = 100


def test_cells_in_shell() -> None:
    assert asoratest.cells_in_shell(0) == 1
    for q in range(1, Q_MAX):
        assert asoratest.cells_in_shell(q) == 4 * q**2 + 2


def test_cells_to_shell() -> None:
    q_tot = 1
    assert asoratest.cells_to_shell(0) == q_tot
    for q in range(1, Q_MAX):
        q_tot += 4 * q**2 + 2
        assert asoratest.cells_to_shell(q) == q_tot


@pytest.mark.parametrize("q", range(0, Q_MAX))
def test_shell_mapping(q: int) -> None:
    cells: set[tuple[int, int, int]] = set()
    q_max = 4 * q**2 + 2 if q > 0 else 1
    for s in range(q_max):
        # Check value makes sense
        ijk = asoratest.linthrd2cart(q, s)
        assert q == sum(abs(x) for x in ijk)

        # Check it's unique
        assert ijk not in cells
        cells.add(ijk)

        # Check inverse function
        assert (q, s) == asoratest.cart2linthrd(*ijk)
