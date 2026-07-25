"""Metallicity-binned photoionization tables for BPASS sources.

Step 1 of wiring metallicity into the radiative transfer.

The GPU raytracer (ASORA) can only hold one photoionization table at a time and
applies it to every source. To make a source's ionizing spectrum depend on its
metallicity, we precompute one table per BPASS metallicity bin up front; the
raytracer then selects the table matching each source's metallicity (Step 4).

This module builds that set of tables. Every table shares the same optical-depth
(``tau``) grid, frequency window, and flux normalization as the rest of the
simulation, so a table for a given Z is identical to the single table the
simulation builds today for that Z.

It replaces an earlier draft that imported a ``StellarPopulationSource`` class
which never existed in the package; the tables are now built directly on the
working :class:`~pyc2ray.radiation.blackbody.BPASSSource`.

Typical use::

    tables = BPASSPhotoTableSet(
        bpass_dir=bpass_dir,
        tau=tau_grid,
        freq_min=ion_freq_HI,
        freq_max=10 * ion_freq_HeII,
        S_star_ref=1e48,
        grey=False,
        freq0=ion_freq_HI,
        pl_index=2.8,
        age=1e7,
    )

    # Per source (Step 4):
    i_bin = tables.bin_index(source_metallicities)        # shape (n_src,)
    thin, thick = tables.get_photo_tables(some_Z)         # (NumTau,), (NumTau,)
"""

import numpy as np
import scipy.integrate
from pathlib import Path
from scipy.optimize import brentq

from .blackbody import (
    BlackBodySource,
    BPASSSource,
    Lsun_erg,
    c_AA,
    h_over_k,
    hplanck,
)

__all__ = [
    "BPASSPhotoTableSet",
    "BBFittedPhotoTableSet",
    "BPASSQionGrid",
    "fit_blackbody_teff",
]


class _ZBinnedTableSet:
    """Shared Z-bin bookkeeping and interpolation for per-metallicity photo
    table sets. Subclasses must set ``Z_bin_centers``, ``photo_thin`` and
    ``photo_thick`` in their ``__init__``."""

    Z_bin_centers: np.ndarray
    photo_thin: np.ndarray
    photo_thick: np.ndarray

    @property
    def n_bins(self):
        """Number of metallicity bins."""
        return self.Z_bin_centers.size

    def bin_index(self, Z):
        """Index of the nearest metallicity bin for each value in ``Z``.

        Parameters
        ----------
        Z : float or array-like
            Metallicity value(s).

        Returns
        -------
        np.ndarray of int, shape (n,)
            The bin index for each input value (always 1-D).
        """
        Z = np.atleast_1d(np.asarray(Z, dtype=float))
        return np.abs(Z[:, None] - self.Z_bin_centers[None, :]).argmin(axis=1)

    def get_photo_tables(self, Z):
        """Return ``(thin, thick)`` tables for the bin nearest to scalar ``Z``.

        Each returned array has shape ``(NumTau,)``.
        """
        i = int(self.bin_index(Z)[0])
        return self.photo_thin[i], self.photo_thick[i]

    def _interp_indices_weight(self, Z):
        """Bracketing bin indices (lo, hi) and log-Z weight w in [0, 1] for scalar Z.

        Z is clamped to the bin range; w = 0 selects bin lo, w = 1 selects bin hi.
        """
        zc = self.Z_bin_centers
        Z = float(np.clip(Z, zc[0], zc[-1]))
        hi = int(np.searchsorted(zc, Z))
        if hi == 0:
            return 0, 0, 0.0
        if hi >= zc.size:
            return zc.size - 1, zc.size - 1, 0.0
        lo = hi - 1
        w = (np.log10(Z) - np.log10(zc[lo])) / (np.log10(zc[hi]) - np.log10(zc[lo]))
        return lo, hi, w

    def get_photo_tables_interp(self, Z):
        """(thin, thick) tables log-linearly interpolated in Z between the two
        bracketing bins. Tables are linear in the SED, so this equals building
        the table from a log-Z-interpolated SED; thick[0] stays S_star_ref."""
        lo, hi, w = self._interp_indices_weight(Z)
        thin = (1.0 - w) * self.photo_thin[lo] + w * self.photo_thin[hi]
        thick = (1.0 - w) * self.photo_thick[lo] + w * self.photo_thick[hi]
        return thin, thick


def fit_blackbody_teff(freqs, sed_photon, nu1, nu2, T_lo=2.0e4, T_hi=1.0e6):
    """Effective temperature of the black body whose photon-SED slope between
    nu1 and nu2 matches the given SED (two-point ratio method, Option 1).

    The BB photon SED is proportional to nu^2 / (exp(h nu / k T) - 1), so the
    ratio SED(nu2)/SED(nu1) is monotonically increasing in T; solve for T with
    brentq. Target ratios outside the achievable BB range (non-thermal slopes)
    return the corresponding temperature bound.
    """
    s1 = float(np.interp(nu1, freqs, sed_photon))
    s2 = float(np.interp(nu2, freqs, sed_photon))
    r_target = s2 / s1

    def bb_ratio(T):
        return (nu2 / nu1) ** 2 * np.expm1(h_over_k * nu1 / T) / np.expm1(h_over_k * nu2 / T)

    if r_target <= bb_ratio(T_lo):
        return T_lo
    if r_target >= bb_ratio(T_hi):
        return T_hi
    return float(brentq(lambda T: bb_ratio(T) - r_target, T_lo, T_hi, xtol=1.0))


class BPASSPhotoTableSet(_ZBinnedTableSet):
    """Photoionization tables for a set of BPASS metallicity bins at one age.

    Parameters
    ----------
    bpass_dir : str or Path
        Directory containing the BPASS SED files.
    tau : array-like
        Optical-depth grid shared with the rest of the simulation. The tables
        have one entry per tau value.
    freq_min, freq_max : float
        Frequency integration window in Hz (same window used to normalize and
        integrate the SED elsewhere in the code).
    S_star_ref : float
        Reference ionizing-photon luminosity each SED is normalized to (1e48).
    grey, freq0, pl_index :
        Passed straight through to ``BPASSSource`` (opacity model and the
        cross-section power-law reference frequency / index).
    age : float
        Stellar-population age in years. A single age is used for all bins;
        adding an age axis later is a straightforward extension.
    metallicities : sequence of float, optional
        Metallicity values to build bins for. Each is snapped to the nearest
        BPASS bin the SED loader supports, then de-duplicated. Defaults to all
        of ``BPASSSource.BPASS_METALLICITIES``.
    log_age_bins : array-like, optional
        log10(age/yr) of the columns in the BPASS files. Passed through to
        ``BPASSSource``.

    Attributes
    ----------
    Z_bin_centers : np.ndarray, shape (NumZ,)
        Sorted metallicity bin centers; row index of ``photo_thin``/``photo_thick``.
    photo_thin, photo_thick : np.ndarray, shape (NumZ, NumTau)
        Optically-thin and optically-thick photoionization tables per bin.
    age : float
        The age (years) the tables were built at.
    """

    def __init__(
        self,
        bpass_dir,
        tau,
        freq_min,
        freq_max,
        S_star_ref,
        grey,
        freq0,
        pl_index,
        age,
        metallicities=None,
        log_age_bins=None,
    ):
        if metallicities is None:
            metallicities = BPASSSource.BPASS_METALLICITIES

        # Snap each requested Z to the nearest bin the BPASS SED loader supports,
        # then de-duplicate and sort so every bin appears exactly once.
        centers = sorted({BPASSSource._snap_metallicity(float(z)) for z in metallicities})
        self.Z_bin_centers = np.array(centers, dtype=float)
        self.age = float(age)

        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        n_z = self.Z_bin_centers.size
        n_tau = tau.size

        self.photo_thin = np.empty((n_z, n_tau), dtype=float)
        self.photo_thick = np.empty((n_z, n_tau), dtype=float)

        for i, Z in enumerate(self.Z_bin_centers):
            src = BPASSSource(Z, self.age, bpass_dir, grey, freq0, pl_index, log_age_bins)
            thin, thick = src.make_photo_table(tau, freq_min, freq_max, S_star_ref)
            self.photo_thin[i] = thin
            self.photo_thick[i] = thick


class BBFittedPhotoTableSet(_ZBinnedTableSet):
    """Option 1: per-Z photoionization tables from black bodies whose T_eff is
    fitted to the HI-edge slope of the corresponding BPASS photon SED.

    Same interface as :class:`BPASSPhotoTableSet` (swap / interpolate by Z);
    only the spectral SHAPE differs. Every table is normalized to the same
    ``S_star_ref``, with amplitudes carried by normflux (q_ion), exactly as for
    the BPASS set — so comparing runs with the two sets isolates the pure
    shape (hardness) effect on the raytracing.
    """

    def __init__(
        self,
        bpass_dir,
        tau,
        freq_min,
        freq_max,
        S_star_ref,
        grey,
        freq0,
        pl_index,
        age,
        metallicities=None,
        log_age_bins=None,
        fit_freq_factor=2.0,
    ):
        if metallicities is None:
            metallicities = BPASSSource.BPASS_METALLICITIES
        centers = sorted({BPASSSource._snap_metallicity(float(z)) for z in metallicities})
        self.Z_bin_centers = np.array(centers, dtype=float)
        self.age = float(age)

        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        n_z, n_tau = self.Z_bin_centers.size, tau.size
        self.photo_thin = np.empty((n_z, n_tau), dtype=float)
        self.photo_thick = np.empty((n_z, n_tau), dtype=float)
        self.Teff = np.empty(n_z, dtype=float)

        for i, Z in enumerate(self.Z_bin_centers):
            src = BPASSSource(Z, self.age, bpass_dir, grey, freq0, pl_index, log_age_bins)
            T = fit_blackbody_teff(
                src.freqs, src.sed_photon, freq_min, fit_freq_factor * freq_min
            )
            self.Teff[i] = T
            bb = BlackBodySource(T, grey, freq0, pl_index)
            thin, thick = bb.make_photo_table(tau, freq_min, freq_max, S_star_ref)
            self.photo_thin[i] = thin
            self.photo_thick[i] = thick


class BPASSQionGrid:
    """q_ion(Z, age): band-integrated ionizing photon rate per SOLAR MASS of a
    BPASS population, tabulated for every (Z bin, age bin).

    This is exactly the amplitude that ``_normalize_sed`` divides out of the
    photo tables (Issue 2): the tables carry pure spectral shape, and this grid
    restores the metallicity- and age-dependent brightness through

        normflux = M_star * q_ion(Z, age) / S_star_ref .

    q_ion(age) already contains the fading of the population (stellar death),
    so runs using it must NOT apply an additional remaining-mass factor.
    """

    def __init__(
        self,
        bpass_dir,
        freq_min,
        freq_max,
        metallicities=None,
        log_age_bins=None,
        population_mass=1e6,
    ):
        if metallicities is None:
            metallicities = BPASSSource.BPASS_METALLICITIES
        centers = sorted({BPASSSource._snap_metallicity(float(z)) for z in metallicities})
        self.Z_bin_centers = np.array(centers, dtype=float)
        if log_age_bins is None:
            log_age_bins = np.round(np.arange(6.0, 11.05, 0.1), decimals=1)
        self.log_age_bins = np.asarray(log_age_bins, dtype=float)
        self.population_mass = float(population_mass)

        n_z, n_age = self.Z_bin_centers.size, self.log_age_bins.size
        self.qion = np.empty((n_z, n_age), dtype=float)
        for i, Z in enumerate(self.Z_bin_centers):
            Z_str = BPASSSource._metallicity_to_str(Z)
            data = np.loadtxt(Path(bpass_dir) / f"spectra-bin-imf135_300.z{Z_str}.dat")
            if data.shape[1] - 1 < n_age:
                raise ValueError(
                    f"BPASS file for Z={Z} has {data.shape[1] - 1} age columns, "
                    f"but {n_age} log_age_bins were requested."
                )
            wl_aa = data[:, 0]
            freqs = c_AA / wl_aa
            order = np.argsort(freqs)
            freqs_sorted = freqs[order]
            band = (freqs_sorted >= freq_min) & (freqs_sorted <= freq_max)
            for j in range(n_age):
                L_lambda = data[:, j + 1]  # L_sun/AA
                L_nu_erg = L_lambda * wl_aa**2 / c_AA * Lsun_erg  # erg/s/Hz
                sed_photon = (L_nu_erg / (hplanck * freqs))[order]  # photons/s/Hz
                self.qion[i, j] = (
                    scipy.integrate.simpson(y=sed_photon[band], x=freqs_sorted[band])
                    / self.population_mass
                )

    @staticmethod
    def _bracket(x, grid):
        """Bracketing indices and linear weight on a sorted 1-D grid, clamped."""
        x = float(np.clip(x, grid[0], grid[-1]))
        hi = int(np.searchsorted(grid, x))
        if hi == 0:
            return 0, 0, 0.0
        if hi >= grid.size:
            return grid.size - 1, grid.size - 1, 0.0
        lo = hi - 1
        w = (x - grid[lo]) / (grid[hi] - grid[lo])
        return lo, hi, w

    def qion_at(self, Z, age_yr):
        """q_ion [photons/s/Msun] bilinearly interpolated in (log Z, log age).

        Z and age are clamped to the tabulated ranges (no extrapolation)."""
        log_zc = np.log10(self.Z_bin_centers)
        zi_lo, zi_hi, wz = self._bracket(np.log10(max(float(Z), 1e-12)), log_zc)
        ai_lo, ai_hi, wa = self._bracket(np.log10(max(float(age_yr), 1.0)), self.log_age_bins)

        q_zlo = (1.0 - wa) * self.qion[zi_lo, ai_lo] + wa * self.qion[zi_lo, ai_hi]
        q_zhi = (1.0 - wa) * self.qion[zi_hi, ai_lo] + wa * self.qion[zi_hi, ai_hi]
        return float((1.0 - wz) * q_zlo + wz * q_zhi)

    def mean_qion_interval(self, Z, t_lo_yr, t_hi_yr, n_sample=256):
        """Interval-averaged ionizing efficiency over age in [t_lo_yr, t_hi_yr]:

            <q_ion> = 1/(t_hi - t_lo) * integral_{t_lo}^{t_hi} q_ion(Z, age) d(age)
                                                                    [photons/s/Msun]

        This is the per-sub-step amplitude: sub-step t spans population ages
        [t*dt, (t+1)*dt], so this returns the average RATE over that window.
        Dividing by the window width (t_hi - t_lo) is essential -- it keeps the
        result a rate (photons/s/Msun), NOT a photon count, so evolve3D's own
        multiplication by dt in the chemistry ODE is not double-counted. q_ion is
        clamped below the youngest tabulated age (1 Myr).
        """
        t_lo, t_hi = float(t_lo_yr), float(t_hi_yr)
        if t_hi <= t_lo:
            return self.qion_at(Z, t_lo)
        ages = np.linspace(t_lo, t_hi, int(n_sample))
        q = np.array([self.qion_at(Z, a) for a in ages])
        return float(scipy.integrate.trapezoid(q, ages) / (t_hi - t_lo))

    def mean_qion(self, Z, t_window_yr, n_sample=256):
        """Whole-slice (0 -> t_window_yr) lifetime average -- a convenience
        wrapper around mean_qion_interval. Kept for the per-slice approach and
        its tests; the per-sub-step driver path uses mean_qion_interval directly.
        """
        return self.mean_qion_interval(Z, 0.0, t_window_yr, n_sample=n_sample)