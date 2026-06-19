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

from .blackbody import BPASSSource

__all__ = ["BPASSPhotoTableSet"]


class BPASSPhotoTableSet:
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