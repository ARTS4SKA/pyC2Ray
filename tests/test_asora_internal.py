from pathlib import Path

import numpy as np
import pytest

try:
    from pyc2ray.lib import libasoratest as asoratest
except ImportError:
    pytest.skip("libasoratest.so missing, skipping tests", allow_module_level=True)


def test_cinterp(data_dir: Path) -> None:
    rng = np.random.default_rng(seed=42)
    N = 11
    dens = rng.random((N, N, N), dtype=np.float64)

    cdens, path = asoratest.cinterp(dens)
    expected_output = np.load(data_dir / "cinterp_output.npz")

    assert np.allclose(cdens, expected_output["cdens"])
    assert np.allclose(path, expected_output["path"])

    expected_output.close()


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
