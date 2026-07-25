"""Tests for the normalisation-scenario machinery: the q_ion(Z, age) grid,
the fitted-Teff black-body table set (Option 1), and the Teff fitter.

Uses the committed mock BPASS SED files under ``radiation/tests/data`` (which
were generated from a T = 1e5 K blackbody, so the Teff fitter has a known
ground truth). CPU-only, runs in seconds.
"""

from pathlib import Path

import numpy as np
import pytest

import astropy.constants as cst
import astropy.units as u

from pyc2ray.radiation.blackbody import BPASSSource
from pyc2ray.radiation.common import make_tau_table
from pyc2ray.radiation.zbinned_tables import (
    BBFittedPhotoTableSet,
    BPASSPhotoTableSet,
    BPASSQionGrid,
    fit_blackbody_teff,
)

LOG_AGE_BINS = np.arange(6.0, 7.1, 0.1)
AGE = 1e7
PL_INDEX = 2.8
METALLICITIES = [0.006, 0.014]


@pytest.fixture(scope="module")
def bpass_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def tau_table():
    tau, _ = make_tau_table(-20.0, 4.0, 200)
    return tau


@pytest.fixture(scope="module")
def freq_range():
    freq_min = (13.598 * u.eV / cst.h).to("Hz").value
    freq_max = 10 * (54.416 * u.eV / cst.h).to("Hz").value
    return freq_min, freq_max


@pytest.fixture(scope="module")
def qion_grid(bpass_dir, freq_range):
    freq_min, freq_max = freq_range
    return BPASSQionGrid(
        bpass_dir=bpass_dir,
        freq_min=freq_min,
        freq_max=freq_max,
        metallicities=METALLICITIES,
        log_age_bins=LOG_AGE_BINS,
    )


# ---------------------------------------------------------------------------
# BPASSQionGrid
# ---------------------------------------------------------------------------

def test_qion_grid_shape_and_positive(qion_grid):
    assert qion_grid.qion.shape == (2, LOG_AGE_BINS.size)
    assert np.all(np.isfinite(qion_grid.qion))
    assert np.all(qion_grid.qion > 0.0)


def test_qion_matches_bpasssource_at_grid_points(qion_grid, bpass_dir, freq_range):
    """A grid entry must equal the un-normalized band integral of the
    corresponding BPASSSource SED (per solar mass)."""
    freq_min, freq_max = freq_range
    for i, Z in enumerate(qion_grid.Z_bin_centers):
        src = BPASSSource(Z, AGE, bpass_dir, False, freq_min, PL_INDEX, LOG_AGE_BINS)
        expected = src.ionizing_photon_rate(freq_min, freq_max) / 1e6
        got = qion_grid.qion_at(Z, AGE)
        np.testing.assert_allclose(got, expected, rtol=1e-8)


def test_qion_interp_bounded_between_bins(qion_grid):
    Z_lo, Z_hi = qion_grid.Z_bin_centers
    Z_mid = np.sqrt(Z_lo * Z_hi)
    q = qion_grid.qion_at(Z_mid, AGE)
    q_lo = qion_grid.qion_at(Z_lo, AGE)
    q_hi = qion_grid.qion_at(Z_hi, AGE)
    assert min(q_lo, q_hi) <= q <= max(q_lo, q_hi)


def test_qion_clamps_out_of_range(qion_grid):
    """Z / age outside the tabulated range clamp to the edges."""
    assert qion_grid.qion_at(1e-10, AGE) == qion_grid.qion_at(
        qion_grid.Z_bin_centers[0], AGE
    )
    assert qion_grid.qion_at(1.0, AGE) == qion_grid.qion_at(
        qion_grid.Z_bin_centers[-1], AGE
    )
    # age below the first bin (1 Myr) clamps to the first bin
    Z = qion_grid.Z_bin_centers[0]
    assert qion_grid.qion_at(Z, 1e5) == qion_grid.qion_at(Z, 10.0 ** LOG_AGE_BINS[0])


# ---------------------------------------------------------------------------
# mean_qion: per-slice lifetime-averaged efficiency
# ---------------------------------------------------------------------------

def test_mean_qion_is_window_average(qion_grid):
    """mean_qion equals a brute-force trapezoid average of qion over [0, T]."""
    import scipy.integrate

    Z = qion_grid.Z_bin_centers[0]
    T_yr = 1e7  # 10 Myr
    qbar = qion_grid.mean_qion(Z, T_yr, n_sample=512)

    ages = np.linspace(0.0, T_yr, 512)
    q = np.array([qion_grid.qion_at(Z, a) for a in ages])
    ref = scipy.integrate.trapezoid(q, ages) / T_yr
    np.testing.assert_allclose(qbar, ref, rtol=1e-12)


def test_mean_qion_bounded_by_window(qion_grid):
    """The lifetime average lies between the youngest (brightest) and oldest
    (faintest) efficiency in the window, and never exceeds the age-0 (peak)
    value."""
    Z = qion_grid.Z_bin_centers[0]
    T_yr = 1e7
    qbar = qion_grid.mean_qion(Z, T_yr)
    q_young = qion_grid.qion_at(Z, 1e6)  # 1 Myr, the youngest tabulated age
    q_old = qion_grid.qion_at(Z, T_yr)   # 10 Myr

    assert min(q_old, q_young) <= qbar <= max(q_old, q_young)
    # age 0 clamps to the 1 Myr (peak) value, so the average can't exceed it
    assert qbar <= q_young * (1.0 + 1e-9)


def test_mean_qion_positive_and_finite(qion_grid):
    for Z in qion_grid.Z_bin_centers:
        qbar = qion_grid.mean_qion(Z, 1e7)
        assert np.isfinite(qbar) and qbar > 0.0


# ---------------------------------------------------------------------------
# mean_qion_interval: per-sub-step interval average
# ---------------------------------------------------------------------------

def test_interval_full_slice_matches_mean_qion(qion_grid):
    """The [0, T] interval average equals the whole-slice mean_qion wrapper."""
    Z = qion_grid.Z_bin_centers[0]
    T = 1e7
    np.testing.assert_allclose(
        qion_grid.mean_qion_interval(Z, 0.0, T),
        qion_grid.mean_qion(Z, T),
        rtol=1e-12,
    )


def test_interval_average_is_bounded(qion_grid):
    """An interval average lies between the efficiency at its two endpoints."""
    Z = qion_grid.Z_bin_centers[0]
    a, b = 2e6, 3e6
    qbar = qion_grid.mean_qion_interval(Z, a, b)
    q_a = qion_grid.qion_at(Z, a)
    q_b = qion_grid.qion_at(Z, b)
    assert min(q_a, q_b) <= qbar <= max(q_a, q_b)


def test_interval_average_is_a_rate_not_a_count(qion_grid):
    """Key units guard: the interval average must be independent of the window
    WIDTH for a locally-constant q_ion (a rate), unlike a bare integral (a count
    that scales with width). Two nested windows starting at the same age, over a
    region where q_ion barely changes, give nearly the same average; their bare
    integrals would differ by the width ratio."""
    Z = qion_grid.Z_bin_centers[0]
    # a region well inside the grid where q_ion varies slowly
    q1 = qion_grid.mean_qion_interval(Z, 5e6, 6e6)
    q2 = qion_grid.mean_qion_interval(Z, 5e6, 5.5e6)
    # both are rates ~ q_ion near 5-6 Myr, so within a modest tolerance
    assert np.isclose(q1, q2, rtol=0.5)
    # and both are the same order as the instantaneous value (a rate)
    assert 0.1 < q1 / qion_grid.qion_at(Z, 5.5e6) < 10.0


def test_substep_intervals_preserve_slice_budget(qion_grid):
    """The whole point of the earlier per-slice approach — total photon budget —
    is preserved by per-sub-step interval averaging: summing each sub-step's
    (interval mean x sub-step width) reproduces the whole-slice (mean x T).

    This is what makes the per-sub-step change safe: it only redistributes
    emission in time within the slice, it does not change the total.
    """
    Z = qion_grid.Z_bin_centers[0]
    T, n = 1e7, 10
    dt = T / n
    total_substep = sum(
        qion_grid.mean_qion_interval(Z, t * dt, (t + 1) * dt, n_sample=512) * dt
        for t in range(n)
    )
    total_slice = qion_grid.mean_qion(Z, T, n_sample=4096) * T
    np.testing.assert_allclose(total_substep, total_slice, rtol=1e-4)


def test_interval_zero_width_returns_instantaneous(qion_grid):
    Z = qion_grid.Z_bin_centers[0]
    assert qion_grid.mean_qion_interval(Z, 4e6, 4e6) == qion_grid.qion_at(Z, 4e6)


# ---------------------------------------------------------------------------
# fit_blackbody_teff
# ---------------------------------------------------------------------------

def test_fit_teff_recovers_mock_blackbody(bpass_dir, freq_range):
    """The committed mock BPASS SEDs are rescaled T = 1e5 K blackbodies, so the
    fitter must recover ~1e5 K from their HI-edge slope."""
    freq_min, _ = freq_range
    src = BPASSSource(0.006, AGE, bpass_dir, False, freq_min, PL_INDEX, LOG_AGE_BINS)
    T = fit_blackbody_teff(src.freqs, src.sed_photon, freq_min, 2.0 * freq_min)
    assert 0.9e5 < T < 1.1e5, f"fitted Teff = {T:.3e} K, expected ~1e5 K"


def test_fit_teff_exact_on_synthetic_blackbody():
    """Fitting a pure synthetic BB photon SED recovers its temperature."""
    from pyc2ray.radiation.blackbody import h_over_k

    T_true = 7.3e4
    nu = np.linspace(1e15, 2e16, 4000)
    sed = nu**2 / np.expm1(h_over_k * nu / T_true)
    T_fit = fit_blackbody_teff(nu, sed, 3.3e15, 6.6e15)
    np.testing.assert_allclose(T_fit, T_true, rtol=1e-3)


# ---------------------------------------------------------------------------
# BBFittedPhotoTableSet (Option 1)
# ---------------------------------------------------------------------------

def test_bb_fitted_tables_interface_and_normalization(bpass_dir, tau_table, freq_range):
    freq_min, freq_max = freq_range
    bb_set = BBFittedPhotoTableSet(
        bpass_dir=bpass_dir,
        tau=tau_table,
        freq_min=freq_min,
        freq_max=freq_max,
        S_star_ref=1e48,
        grey=False,
        freq0=freq_min,
        pl_index=PL_INDEX,
        age=AGE,
        metallicities=METALLICITIES,
        log_age_bins=LOG_AGE_BINS,
    )
    n_tau = np.atleast_1d(tau_table).size
    assert bb_set.photo_thin.shape == (2, n_tau)
    assert bb_set.photo_thick.shape == (2, n_tau)
    assert np.all(np.isfinite(bb_set.Teff))
    # every table shares the reference normalization at tau = 0
    for i in range(bb_set.n_bins):
        assert np.isclose(bb_set.photo_thick[i, 0], 1e48, rtol=1e-3)
    # interpolation interface inherited from the shared base class
    thin, thick = bb_set.get_photo_tables_interp(np.sqrt(0.006 * 0.014))
    assert thin.shape == (n_tau,)
    assert np.isclose(thick[0], 1e48, rtol=1e-3)


def test_bb_fitted_tables_match_bpass_tables_on_mock(bpass_dir, tau_table, freq_range):
    """The mock BPASS SEDs ARE blackbodies, so Option 1's fitted-BB tables must
    closely reproduce the BPASS-shape tables on this data (real BPASS data
    would legitimately differ — that difference is the point of the comparison)."""
    freq_min, freq_max = freq_range
    common = dict(
        bpass_dir=bpass_dir,
        tau=tau_table,
        freq_min=freq_min,
        freq_max=freq_max,
        S_star_ref=1e48,
        grey=False,
        freq0=freq_min,
        pl_index=PL_INDEX,
        age=AGE,
        metallicities=METALLICITIES,
        log_age_bins=LOG_AGE_BINS,
    )
    bb_set = BBFittedPhotoTableSet(**common)
    bpass_set = BPASSPhotoTableSet(**common)
    np.testing.assert_allclose(bb_set.photo_thick, bpass_set.photo_thick, rtol=0.05)
    np.testing.assert_allclose(bb_set.photo_thin, bpass_set.photo_thin, rtol=0.05)
