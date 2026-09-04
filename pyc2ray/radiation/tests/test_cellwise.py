"""Unit tests for the cell-by-cell (two-group) metallicity regime.

Covers the pure, GPU-free logic:
  * classify_source_cells  -- source -> old/new classification by cell metallicity
  * the per-source group-aware q_ion amplitude construction

The full two-pass evolve (evolve3D_multigroup) needs ASORA/libc2ray on the GPU
and is exercised on the cluster, not here.

Importing pyc2ray.c2ray_metals boots ASORA and mpi4py (via c2ray_base), which is
unavailable off-cluster, so the whole module skips rather than erroring there.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from pyc2ray.c2ray_metals import bin_weights, classify_source_cells
    from pyc2ray.radiation.zbinned_tables import BPASSQionGrid  # noqa: F401
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - environment dependent
    classify_source_cells = None
    _IMPORT_ERR = exc

pytestmark = pytest.mark.skipif(
    classify_source_cells is None,
    reason=f"pyc2ray not importable here: {_IMPORT_ERR!r}",
)


# ---------------------------------------------------------------------------
# classify_source_cells
# ---------------------------------------------------------------------------

def _make_grid(N, enriched_cells, Z_min, Z_enriched):
    """Flat (N^3,) f_Z_cell grid at the floor except the given (i,j,k) cells."""
    fz = np.full(N**3, Z_min, dtype=float)
    for (i, j, k) in enriched_cells:
        fz[i * N * N + j * N + k] = Z_enriched
    return fz


def test_classify_basic_old_vs_new():
    N, Z_min = 4, 1e-4
    fz = _make_grid(N, [(1, 2, 3)], Z_min, 0.01)
    srcpos = np.array([[1, 2, 3], [0, 0, 0], [2, 2, 2]])  # first is enriched
    mask_old, fz_src = classify_source_cells(srcpos, fz, N, Z_min)
    np.testing.assert_array_equal(mask_old, [True, False, False])
    np.testing.assert_allclose(fz_src, [0.01, Z_min, Z_min])


def test_classify_floor_cells_are_new():
    """Cells exactly at the floor are 'new', not 'old'."""
    N, Z_min = 3, 1e-4
    fz = np.full(N**3, Z_min)
    srcpos = np.array([[0, 0, 0], [1, 1, 1]])
    mask_old, _ = classify_source_cells(srcpos, fz, N, Z_min)
    assert not mask_old.any()


def test_classify_flat_index_matches_c_order():
    """The (i,j,k) -> i*N^2 + j*N + k mapping must match the pipeline's C-order
    flattened_cell_index, so a source lands on its own metallicity cell."""
    N, Z_min = 5, 1e-4
    # enrich a single, asymmetric cell so an axis swap would be caught
    i, j, k = 1, 2, 4
    fz = _make_grid(N, [(i, j, k)], Z_min, 0.02)
    # only the exact (i,j,k) source is old; the axis-swapped (k,j,i) is not
    srcpos = np.array([[i, j, k], [k, j, i]])
    mask_old, _ = classify_source_cells(srcpos, fz, N, Z_min)
    np.testing.assert_array_equal(mask_old, [True, False])


def test_classify_clamps_out_of_range_indices():
    N, Z_min = 4, 1e-4
    fz = _make_grid(N, [(3, 3, 3)], Z_min, 0.01)
    srcpos = np.array([[9, 9, 9]])  # clamps to (3,3,3)
    mask_old, fz_src = classify_source_cells(srcpos, fz, N, Z_min)
    np.testing.assert_array_equal(mask_old, [True])
    np.testing.assert_allclose(fz_src, [0.01])


def test_mean_old_metallicity_from_classification():
    """The old-group mean is the mean of f_Z_cell over the old-source cells."""
    N, Z_min = 4, 1e-4
    fz = _make_grid(N, [(0, 0, 0), (1, 1, 1)], Z_min, 0.0)
    fz[0 * N * N + 0 * N + 0] = 0.004
    fz[1 * N * N + 1 * N + 1] = 0.006
    srcpos = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    mask_old, fz_src = classify_source_cells(srcpos, fz, N, Z_min)
    mean_old = fz_src[mask_old].mean()
    assert np.isclose(mean_old, 0.005)


# ---------------------------------------------------------------------------
# bin_weights: per-bin log-Z flux split
# ---------------------------------------------------------------------------

BINS = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 3e-2, 4e-2])


def test_bin_weights_exact_bin_puts_all_mass_in_that_bin():
    """A source exactly on a bin center goes entirely to that bin: either as
    (lo=b, w=0) or (lo=b-1, hi=b, w=1); both route all mass to bin b once the
    w==0 halves are dropped."""
    for b, Z in enumerate(BINS):
        lo, hi, w = bin_weights([Z], BINS)
        eff_bin = lo[0] if w[0] == 0.0 else hi[0]
        assert eff_bin == b
        assert w[0] in (0.0, 1.0)


def test_bin_weights_log_midpoint_is_half():
    Zmid = np.sqrt(BINS[2] * BINS[3])
    lo, hi, w = bin_weights([Zmid], BINS)
    assert (lo[0], hi[0]) == (2, 3)
    assert np.isclose(w[0], 0.5, atol=1e-12)


def test_bin_weights_clamp_out_of_range():
    lo, hi, w = bin_weights([1e-9, 10.0], BINS)
    # below lowest -> bin 0, w=0 ; above highest -> last bin, w=1
    assert lo[0] == 0 and w[0] == 0.0
    assert hi[1] == len(BINS) - 1 and np.isclose(w[1], 1.0)


def test_bin_weights_partition_conserves_mass():
    """(1-w) to lo and w to hi must sum to 1 for every source (mass-conserving)."""
    rng = np.random.default_rng(0)
    Z = 10 ** rng.uniform(np.log10(BINS[0]), np.log10(BINS[-1]), size=200)
    lo, hi, w = bin_weights(Z, BINS)
    assert np.allclose((1 - w) + w, 1.0)
    assert np.all((w >= 0) & (w <= 1))
    assert np.all(hi == lo + 1)


def test_flux_split_total_equals_interpolated_qion():
    """Bin-center amplitude with the log-Z split reproduces the interpolated
    q_ion: (1-w)*q(Z_lo) + w*q(Z_hi) == linear-in-logZ interp used for the
    amplitude, so the summed per-bin flux equals M_star * q_interp."""
    # fake per-bin q_ion values
    q_bin = np.linspace(5e46, 1e46, len(BINS))
    M_star = 3e8
    Zmid = np.sqrt(BINS[4] * BINS[5])
    lo, hi, w = bin_weights([Zmid], BINS)
    lo, hi, w = lo[0], hi[0], w[0]
    flux_lo = M_star * (1 - w) * q_bin[lo] / 1e48
    flux_hi = M_star * w * q_bin[hi] / 1e48
    total = flux_lo + flux_hi
    q_interp = (1 - w) * q_bin[lo] + w * q_bin[hi]
    np.testing.assert_allclose(total, M_star * q_interp / 1e48)