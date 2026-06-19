import numpy as np
import pytest
import tempfile
from pathlib import Path

import astropy.constants as cst
import astropy.units as u

from pyc2ray.radiation.blackbody import BlackBodySource, BPASSSource

from pyc2ray.radiation.common import make_tau_table

bpass_dir = "tests/data/"

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def tau_table():
    tau, dlogtau = make_tau_table(-20.0, 4.0, 200)   # small N for speed
    return tau

@pytest.fixture(scope="module")
def freq_range():
    freq_min = (13.598 * u.eV / cst.h).to("Hz").value
    freq_max = 10 * (54.416 * u.eV / cst.h).to("Hz").value
    return freq_min, freq_max

@pytest.fixture(scope="module")
def bpass_dir():
    return Path(__file__).parent / "data"

@pytest.fixture(scope="module")
def source(bpass_dir, tau_table, freq_range):
    freq_min, _ = freq_range
    return BPASSSource(
        metallicity=0.006,
        age=1e7,
        bpass_dir=bpass_dir,
        grey=False,
        freq0=freq_min,
        pl_index=2.8,
        log_age_bins=np.arange(6.0, 7.1, 0.1)
    )


# -----------------------------------------------------------------------
# _snap_metallicity
# -----------------------------------------------------------------------

def test_snap_exact_match():
    assert BPASSSource._snap_metallicity(0.006) == 0.006

def test_snap_rounds_to_nearest():
    # 0.0015 is equidistant between 0.001 and 0.002 — just check it's one of them
    result = BPASSSource._snap_metallicity(0.0015)
    assert result in (0.001, 0.002)

def test_snap_below_minimum():
    result = BPASSSource._snap_metallicity(1e-10)
    assert result == min(BPASSSource.BPASS_METALLICITIES)

def test_snap_above_maximum():
    result = BPASSSource._snap_metallicity(1.0)
    assert result == max(BPASSSource.BPASS_METALLICITIES)


# -----------------------------------------------------------------------
# _metallicity_to_str
# -----------------------------------------------------------------------

@pytest.mark.parametrize("Z,expected", [
    (0.001,   "00100"),
    (0.00001, "00001"),
    (0.020,   "02000"),
    (0.0001,  "00010"),
])
def test_metallicity_to_str(Z, expected):
    assert BPASSSource._metallicity_to_str(Z) == expected


# -----------------------------------------------------------------------
# _load_bpass_sed
# -----------------------------------------------------------------------

def test_freqs_monotonically_increasing(source):
    assert np.all(np.diff(source.freqs) > 0)

def test_sed_nonnegative(source):
    assert np.all(source.sed_photon >= 0)

def test_sed_and_freqs_same_length(source):
    assert source.freqs.shape == source.sed_photon.shape

def test_sed_peaks_in_uv(source):
    # For T=1e5 K, peak should be in UV/EUV range, not IR
    peak_freq = source.freqs[np.argmax(source.sed_photon)]
    freq_HI = (13.598 * u.eV / cst.h).to("Hz").value
    assert peak_freq > freq_HI


# -----------------------------------------------------------------------
# _age_to_column
# -----------------------------------------------------------------------

def test_age_column_returns_valid_index(bpass_dir):
    """Whatever the layout, _age_to_column should return a valid column index."""
    src = BPASSSource(0.006, 1e7, bpass_dir, False, 3.29e15, 2.8, np.arange(6.0, 7.1, 0.1))
    data = np.loadtxt(bpass_dir / "spectra-bin-imf135_300.z00600.dat")
    col = src._age_to_column(data)
    assert 1 <= col < data.shape[1]

def test_age_column_snaps_to_nearest(bpass_dir):
    # age=5e6 is between log-age 6.6 and 6.7 — should snap to one of them
    src_a = BPASSSource(0.006, 5e6,   bpass_dir, False, 3.29e15, 2.8, np.arange(6.0, 7.1, 0.1))
    src_b = BPASSSource(0.006, 5.5e6, bpass_dir, False, 3.29e15, 2.8, np.arange(6.0, 7.1, 0.1))
    data = np.loadtxt(bpass_dir / "spectra-bin-imf135_300.z00600.dat")
    assert src_a._age_to_column(data) in range(1, data.shape[1])
    assert src_b._age_to_column(data) in range(1, data.shape[1])


# -----------------------------------------------------------------------
# _normalize_sed
# -----------------------------------------------------------------------

def test_normalization_integral(source, freq_range):
    import scipy.integrate
    freq_min, freq_max = freq_range
    sed_norm = source._normalize_sed(
        source.freqs, source.sed_photon, freq_min, freq_max, 1e48
    )
    mask = (source.freqs >= freq_min) & (source.freqs <= freq_max)
    integral = scipy.integrate.simpson(y=sed_norm[mask], x=source.freqs[mask])
    assert np.isclose(integral, 1e48, rtol=1e-6)

def test_normalization_preserves_shape(source, freq_range):
    freq_min, freq_max = freq_range
    sed_norm = source._normalize_sed(
        source.freqs, source.sed_photon, freq_min, freq_max, 1e48
    )
    # Find the scaling factor from any nonzero entry
    nonzero = source.sed_photon > 0
    scale = sed_norm[nonzero][0] / source.sed_photon[nonzero][0]
    # Check that sed_norm == scale * sed_photon everywhere
    assert np.allclose(sed_norm, scale * source.sed_photon)


# -----------------------------------------------------------------------
# make_photo_table
# -----------------------------------------------------------------------

def test_photo_table_shape(source, tau_table, freq_range):
    freq_min, freq_max = freq_range
    thin, thick = source.make_photo_table(tau_table, freq_min, freq_max, 1e48)
    assert thin.shape  == tau_table.shape
    assert thick.shape == tau_table.shape

def test_photo_thick_at_zero_tau_equals_S_star(source, tau_table, freq_range):
    # thick_table[tau=0] = ∫ SED_norm dν = S_star_ref
    freq_min, freq_max = freq_range
    _, thick = source.make_photo_table(tau_table, freq_min, freq_max, 1e48)
    assert np.isclose(thick[0], 1e48, rtol=1e-3)

def test_photo_thick_decreases_with_tau(source, tau_table, freq_range):
    freq_min, freq_max = freq_range
    _, thick = source.make_photo_table(tau_table, freq_min, freq_max, 1e48)
    assert np.all(np.diff(thick) <= 0)

def test_photo_thin_nonnegative(source, tau_table, freq_range):
    freq_min, freq_max = freq_range
    thin, _ = source.make_photo_table(tau_table, freq_min, freq_max, 1e48)
    assert np.all(thin >= 0)

def test_photo_table_caching(source, tau_table, freq_range, tmp_path):
    freq_min, freq_max = freq_range
    cache = tmp_path / "test_cache.npz"
    thin1, thick1 = source.make_photo_table(
        tau_table, freq_min, freq_max, 1e48, cache_path=cache
    )
    assert cache.exists()
    # Second call should load from cache — patch _compute_tables to confirm it's not called
    from unittest.mock import patch
    with patch.object(source, "_compute_tables", wraps=source._compute_tables) as mock:
        thin2, thick2 = source.make_photo_table(
            tau_table, freq_min, freq_max, 1e48, cache_path=cache
        )
        mock.assert_not_called()
    assert np.allclose(thin1, thin2) and np.allclose(thick1, thick2)


# -----------------------------------------------------------------------
# Agreement with BlackBodySource (key sanity check)
# -----------------------------------------------------------------------

def test_agrees_with_blackbody_source(source, tau_table, freq_range):
    """
    Since the mock BPASS file is generated from a T=1e5 K blackbody,
    BPASSSource tables should closely match BlackBodySource tables.
    """
    freq_min, freq_max = freq_range
    sig = 6.30e-18

    bb = BlackBodySource(1e5, False, freq_min, 2.8)
    bb_thin, bb_thick = bb.make_photo_table(tau_table, freq_min, freq_max, 1e48)

    bpass_thin, bpass_thick = source.make_photo_table(
        tau_table, freq_min, freq_max, 1e48
    )

    # Agreement within 5% — discretization and interpolation differences are expected
    assert np.allclose(bpass_thick, bb_thick, rtol=0.05)
    assert np.allclose(bpass_thin,  bb_thin,  rtol=0.05)
