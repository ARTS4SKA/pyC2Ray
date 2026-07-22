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
