from abc import ABC, abstractmethod
from functools import partial
from typing import TypeVar

import astropy.constants as cst
import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy
from scipy.integrate import quad, quad_vec

from pathlib import Path

from ..utils.sourceutils import PathType

import pyc2ray as pc2r

# For detailed comparisons with C2Ray, we use the same exact value for the constants
# This can be changed to the astropy values once consistency between
# the two codes has been established
h_over_k = (cst.h / cst.k_B).cgs.value
pi = np.pi
c = cst.c.cgs.value
two_pi_over_c_square = 2.0 * pi / (c * c)
hplanck = cst.h.cgs.value
sigma_0 = 6.3e-18

# CGS constants
Lsun_erg   = cst.L_sun.cgs.value
c_cgs      = cst.c.cgs.value
c_AA       = cst.c.to(u.AA / u.s).value   # speed of light in Å/s
ion_freq_HI = (cst.Ryd * cst.c).cgs.value # Hz

__all__ = ["BlackBodyBase", "BPASSSource", "BlackBodySource", "YggdrasilModel"]

BlackBodyType = TypeVar("BlackBodyType", bound="BlackBodyBase")

FloatArray = npt.NDArray[np.float64]


class BlackBodyBase(ABC):
    @abstractmethod
    def make_photo_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]: ...

    @abstractmethod
    def make_heat_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]: ...


class BPASSSource(BlackBodyBase):
    """
    Point source whose spectral shape is a blackbody normalised using
    BPASS stellar population data.

    BPASS provides one file per metallicity: rows = wavelength (Angstrom),
    columns = log-age bins. For a given (metallicity, age), this class
    derives applies the user-supplied normalization to match the BPASS 
    SED and ionizing photon rate.

    Assumes that the SED files are already normalised
    """

    # TO DO: Change to access directly from parameter file instead (?)
    # The full BPASS v2.2.1 grid: must match the 13 entries of the BPASS
    # distribution's metals.txt (which BPASSYieldTable reads directly), so that
    # the spectral / q_ion bins and the yield bins share one metallicity grid.
    BPASS_METALLICITIES: list[float] = [
        0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.006,
        0.008, 0.01, 0.014, 0.02, 0.03, 0.04,
    ]
    
    def __init__(
        self,
        metallicity: float,
        age: float,
        bpass_dir: PathType,
        grey: bool,
        freq0: float,
        pl_index: float,
        log_age_bins: FloatArray | None = None,
    ) -> None:

        # if not provided, assume full-resolution BPASS (52 columns)
        if log_age_bins is None:
            self.log_age_bins = np.round(np.arange(6.0, 11.05, 0.1), decimals=1)
        else:
            self.log_age_bins = np.asarray(log_age_bins)
            
        self.grey = grey
        self.freq0 = freq0
        self.pl_index = pl_index
        self.metallicity = self._snap_metallicity(metallicity)
        self.age = age

        # Load BPASS SED for this metallicity and age
        self.freqs, self.sed_photon = self._load_bpass_sed(Path(bpass_dir))


    # ------------------------------------------------------------------
    # Metallicity snapping and helpers
    # ------------------------------------------------------------------

    @classmethod
    def _snap_metallicity(cls, Z: float) -> float:
        """Return the nearest BPASS metallicity bin to Z."""
        return min(cls.BPASS_METALLICITIES, key=lambda z: abs(z - Z))

    @staticmethod
    def _metallicity_to_str(Z: float) -> str:
        """
        Convert metallicity float to BPASS filename suffix.
        e.g. 0.001 → '00100',  0.00001 → '00001'
        Matches the convention: ('%.5f' % Z)[2:]
        """
        return ('%.5f' % Z)[2:]

    # ------------------------------------------------------------------
    # BPASS table loading
    # ------------------------------------------------------------------

    def _load_bpass_sed(
        self, bpass_dir: PathType
    ) -> tuple[FloatArray, FloatArray]:
        """
        Load BPASS table for self.metallicity and return the SED at self.age.

        Expected BPASS file layout:
            Column 0  : wavelength (Angstrom)
            Columns 1+: L_lambda (L_sun/Å) at each log-age bin

        Returns
        -------
        freqs      : frequencies in Hz, monotonically increasing
        sed_photon : photon-rate SED in photons/s/Hz
                     (un-normalised; normalisation happens in make_photo_table)
        """
        
        # TODO: adapt path convention to BPASS version
        # e.g. "spectra-bin-imf135_300.z{Z_str}.dat"
        Z_str    = self._metallicity_to_str(self.metallicity)
        filepath = bpass_dir / f"spectra-bin-imf135_300.z{Z_str}.dat"

        data     = np.loadtxt(filepath)
        wl_aa    = data[:, 0]                       # Angstrom
        age_col  = self._age_to_column(data)
        L_lambda = data[:, age_col]                 # L_sun/Å

        # Wavelength → frequency; ensure monotonically increasing
        freqs = c_AA / wl_aa
        if freqs[0] > freqs[-1]:
            freqs    = freqs[::-1]
            L_lambda = L_lambda[::-1]
            wl_aa    = wl_aa[::-1]
        
        # L_lambda [L_sun/Å] → L_nu [L_sun/Hz]:  L_nu = L_lambda × lambda² / c
        L_nu_Lsun  = L_lambda * wl_aa**2 / c_AA    # L_sun/Hz

        # L_nu [L_sun/Hz] → photon SED [photons/s/Hz]:  divide by h*nu
        L_nu_erg   = L_nu_Lsun * Lsun_erg          # erg/s/Hz
        sed_photon = L_nu_erg / (hplanck * freqs)   # photons/s/Hz

        return freqs, sed_photon 

    # TO DO: Interpolate between BPASS ages, or snap to nearest age?
    def _age_to_column(self, data: FloatArray) -> int:
        log_age_target = np.log10(self.age)
        idx = int(np.argmin(np.abs(self.log_age_bins - log_age_target)))
        return idx + 1   # +1 for wavelength column

    # ------------------------------------------------------------------
    # Cross-section frequency dependence
    # ------------------------------------------------------------------

    # TO DO: check formula
    def _cross_section(self, freq: FloatArray) -> FloatArray:
        if self.grey:
            return np.ones_like(freq)
        return (freq / self.freq0) ** (-self.pl_index)

    # ------------------------------------------------------------------
    # SED normalisation
    # ------------------------------------------------------------------

    def _normalize_sed(
        self,
        freqs: FloatArray,
        sed: FloatArray,
        freq_min: float,
        freq_max: float,
        S_star_ref: float,
    ) -> FloatArray:
        """
        Scale the photon-rate SED so that
            ∫_{freq_min}^{freq_max} SED dν  =  S_star_ref.
        Returns the full (all-frequency) scaled SED.
        """
        mask      = (freqs >= freq_min) & (freqs <= freq_max)
        S_unscaled = scipy.integrate.simpson(y=sed[mask], x=freqs[mask])
        return sed * (S_star_ref / S_unscaled)

    def ionizing_photon_rate(self, freq_min: float, freq_max: float) -> float:
        """Un-normalized band-integrated ionizing photon rate Q_ion for this
        (metallicity, age) BPASS population, in photons/s per BPASS population
        mass (the loader's 1e6 Msun normalization).

        This is exactly the S_unscaled that _normalize_sed divides out — the
        metallicity/age-dependent ionizing EFFICIENCY that make_photo_table
        normalizes away. Ratios between metallicities are unit-independent."""
        mask = (self.freqs >= freq_min) & (self.freqs <= freq_max)
        return float(scipy.integrate.simpson(y=self.sed_photon[mask], x=self.freqs[mask]))

    # ------------------------------------------------------------------
    # Integrand helpers
    # ------------------------------------------------------------------

    def _photo_thick_integrand(
        self, freqs: FloatArray, tau: float, sed_norm: FloatArray
    ) -> FloatArray:
        sigma   = self._cross_section(freqs)
        exponent = tau * sigma
        return np.where(exponent < 700.0, sed_norm * np.exp(-exponent), 0.0)

    def _photo_thin_integrand(
        self, freqs: FloatArray, tau: float, sed_norm: FloatArray
    ) -> FloatArray:
        sigma    = self._cross_section(freqs)
        exponent = tau * sigma
        return np.where(exponent < 700.0, sed_norm * sigma * np.exp(-exponent), 0.0)

    def _heat_thick_integrand(
        self, freqs: FloatArray, tau: float, sed_norm: FloatArray
    ) -> FloatArray:
        excess_energy = hplanck * (freqs - ion_freq_HI)
        return excess_energy * self._photo_thick_integrand(freqs, tau, sed_norm)

    def _heat_thin_integrand(
        self, freqs: FloatArray, tau: float, sed_norm: FloatArray
    ) -> FloatArray:
        excess_energy = hplanck * (freqs - ion_freq_HI)
        return excess_energy * self._photo_thin_integrand(freqs, tau, sed_norm)

    # ------------------------------------------------------------------
    # Core integration loop
    # ------------------------------------------------------------------

    def _compute_tables(
        self,
        tau: FloatArray,
        freq_min: float,
        freq_max: float,
        S_star_ref: float,
        thin_integrand,
        thick_integrand,
    ) -> tuple[FloatArray, FloatArray]:
        sed_norm  = self._normalize_sed(
            self.freqs, self.sed_photon, freq_min, freq_max, S_star_ref
        )
        mask      = (self.freqs >= freq_min) & (self.freqs <= freq_max)
        freqs_r   = self.freqs[mask]
        sed_r     = sed_norm[mask]

        table_thin  = np.array([
            scipy.integrate.simpson(y=thin_integrand(freqs_r, t, sed_r),  x=freqs_r)
            for t in tau
        ])
        table_thick = np.array([
            scipy.integrate.simpson(y=thick_integrand(freqs_r, t, sed_r), x=freqs_r)
            for t in tau
        ])
        return table_thin, table_thick

    # ------------------------------------------------------------------
    # BlackBodyBase interface
    # ------------------------------------------------------------------

    def make_photo_table(
        self,
        tau: FloatArray,
        freq_min: float,
        freq_max: float,
        S_star_ref: float,
        cache_path: PathType | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        if cache_path is not None and Path(cache_path).exists():
            data = np.load(cache_path)
            return data["thin"], data["thick"]

        thin, thick = self._compute_tables(
            tau, freq_min, freq_max, S_star_ref,
            self._photo_thin_integrand, self._photo_thick_integrand,
        )

        if cache_path is not None:
            np.savez(cache_path, thin=thin, thick=thick,
                     metallicity=self.metallicity, age=self.age)
        return thin, thick

    def make_heat_table(
        self,
        tau: FloatArray,
        freq_min: float,
        freq_max: float,
        S_star_ref: float,
        cache_path: PathType | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        if cache_path is not None and Path(cache_path).exists():
            data = np.load(cache_path)
            return data["thin"], data["thick"]

        thin, thick = self._compute_tables(
            tau, freq_min, freq_max, S_star_ref,
            self._heat_thin_integrand, self._heat_thick_integrand,
        )

        if cache_path is not None:
            np.savez(cache_path, thin=thin, thick=thick,
                     metallicity=self.metallicity, age=self.age)
        return thin, thick

    # ------------------------------------------------------------------
    # Pre-computation over all (Z, age) combinations
    # ------------------------------------------------------------------

    @classmethod
    def precompute_tables(
        cls,
        metallicities: list[float],
        ages: list[float],
        bpass_dir: PathType,
        tau: FloatArray,
        freq_min: float,
        freq_max: float,
        S_star_ref: float,
        cache_dir: PathType,
        grey: bool,
        freq0: float,
        pl_index: float,
        log_age_bins: FloatArray | None = None,
        compute_heat: bool = False,
    ) -> dict[tuple[float, float], tuple[FloatArray, FloatArray]]:
        """
        Pre-compute and cache photo (and optionally heat) tables for every
        (metallicity, age) pair. Skips pairs whose cache file already exists.
    
        Parameters
        ----------
        log_age_bins : array of log10(age/yr) values corresponding to the columns
            in the BPASS files (after the wavelength column). If None, assumes the
            full-resolution BPASS layout (log_age 6.0 to 11.0 in 0.1 dex steps).
    
        Returns
        -------
        dict keyed by (Z, age) → (thin_table, thick_table)
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tables = {}
    
        for Z in metallicities:
            for age in ages:
                src = cls(Z, age, bpass_dir, grey, freq0, pl_index, log_age_bins)
                photo_path = cache_dir / f"photo_Z{Z:.5f}_age{age:.3e}.npz"
                tables[(Z, age)] = src.make_photo_table(
                    tau, freq_min, freq_max, S_star_ref, cache_path=photo_path
                )
                if compute_heat:
                    heat_path = cache_dir / f"heat_Z{Z:.5f}_age{age:.3e}.npz"
                    src.make_heat_table(
                        tau, freq_min, freq_max, S_star_ref, cache_path=heat_path
                    )
    
        return tables

    

class BlackBodySource(BlackBodyBase):
    """A point source emitting a Black-body spectrum"""

    def __init__(self, temp: float, grey: bool, freq0: float, pl_index: float) -> None:
        self.temp = temp
        self.grey = grey
        self.freq0 = freq0
        self.pl_index = pl_index
        self.R_star = 1.0

    def SED(self, freq: float) -> float:
        if freq * h_over_k / self.temp >= 700.0:
            return 0.0
        return (
            4
            * np.pi
            * self.R_star**2
            * two_pi_over_c_square
            * freq**2
            / (np.exp(freq * h_over_k / self.temp) - 1.0)
        )

    def integrate_SED(self, f1: float, f2: float) -> float:
        res, *_ = quad(self.SED, f1, f2)
        return res

    def normalize_SED(self, f1: float, f2: float, S_star_ref: float) -> None:
        S_unscaled = self.integrate_SED(f1, f2)
        S_scaling = S_star_ref / S_unscaled
        self.R_star = np.sqrt(S_scaling) * self.R_star

    def cross_section_freq_dependence(self, freq: float) -> float:
        if self.grey:
            return 1.0
        return (freq / self.freq0) ** (-self.pl_index)

    # C2Ray distinguishes between optically thin and thick cells,
    # and calculates the rates differently for those two cases.
    # See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        itg = self.SED(freq) * np.exp(-tau * self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _photo_thin_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        itg = (
            self.SED(freq)
            * self.cross_section_freq_dependence(freq)
            * np.exp(-tau * self.cross_section_freq_dependence(freq))
        )
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _heat_thick_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        photo_thick = self._photo_thick_integrand_vec(freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thick

    def _heat_thin_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        photo_thin = self._photo_thin_integrand_vec(freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thin

    def make_photo_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]:
        self.normalize_SED(freq_min, freq_max, S_star_ref)

        integrand_thin = partial(self._photo_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._photo_thick_integrand_vec, tau=tau)

        table_thin = quad_vec(integrand_thin, freq_min, freq_max, epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick, freq_min, freq_max, epsrel=1e-12)[0]
        return table_thin, table_thick

    def make_heat_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]:
        self.normalize_SED(freq_min, freq_max, S_star_ref)

        integrand_thin = partial(self._heat_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._heat_thick_integrand_vec, tau=tau)

        table_thin = quad_vec(integrand_thin, freq_min, freq_max, epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick, freq_min, freq_max, epsrel=1e-12)[0]
        return table_thin, table_thick


class YggdrasilModel(BlackBodyBase):
    """Use Yggdrasil model for SED"""

    def __init__(
        self, tabname: str, grey: bool, freq0: float, pl_index: float, S_star_ref: float
    ) -> None:
        self.grey = grey
        self.freq0 = freq0
        self.tabname = tabname
        self.pl_index = pl_index

    """
    # This was used for debugging. Can be usefull in the future(?)
    def SED(self, f1, f2):
        freqs = np.linspace(f2, f1, 10) * u.Hz
        lamb = (cst.c  / freqs).to('AA')
        R_star, temp = 1*u.Rsun, 5e4*u.K

        #ampl = 8*np.pi**2 * R_star**2 *cst.h * cst.c**2 / lamb**5
        #sed = ampl / (np.exp((cst.h*cst.c/lamb/cst.k_B/temp).cgs)-1.0)
        ampl = 8*np.pi**2 * R_star**2 *cst.h * freqs**5/cst.c**3
        sed = ampl / (np.exp(freqs*cst.h/cst.k_B/temp)-1.0)
        
        return sed.to('erg / s / AA').value, freqs.value, lamb.value
        
    """

    def SED(self, f1: float, f2: float) -> tuple[float, FloatArray, FloatArray]:
        lamb, flux = np.loadtxt(
            self.tabname, unpack=True
        )  # wavelenght in (Angstrom), Flux in (erg/s/AA)
        freqs = (cst.c / (lamb * u.AA)).to("Hz").value

        if (freqs.min() == freqs[-1]) and (freqs.max() == freqs[0]):
            # this is for the discrete integral scipy.integrate.simpson, which require increasing value for the x-axis
            lamb = lamb[::-1]
            freqs = freqs[::-1]
            flux = flux[::-1]

        int_range = (freqs >= f1) * (freqs <= f2)
        sed = flux[int_range]
        return sed, freqs[int_range], lamb[int_range]

    def integrate_SED(self, sed: float, freq: FloatArray) -> float:
        assert freq.min() == freq[0]
        assert freq.max() == freq[-1]

        return scipy.integrate.simpson(x=freq, y=sed)

    def normalize_SED(self, sed: float, freq: FloatArray, S_star_ref: float) -> float:
        S_unscaled = self.integrate_SED(sed, freq)
        # MB: in C2Ray this was: self.R_star = np.sqrt(S_scaling) * self.R_star. Here we define the SED with the proper units so we do not need to squareroot (as we do not multiply to R_star) and instead multiply directly to the SED.
        S_scaling = S_star_ref / S_unscaled
        return sed * S_scaling

    def cross_section_freq_dependence(self, freq: FloatArray) -> FloatArray:
        if self.grey:
            return np.ones_like(freq)
        return (freq / self.freq0) ** (-self.pl_index)

    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        itg = sed * np.exp(-tau * self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _photo_thin_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        itg = (
            sed
            * self.cross_section_freq_dependence(freq)
            * np.exp(-tau * self.cross_section_freq_dependence(freq))
        )
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _heat_thick_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        photo_thick = self._photo_thick_integrand_vec(sed, freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thick

    def _heat_thin_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        photo_thin = self._photo_thin_integrand_vec(sed, freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thin

    def make_photo_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]:
        sed, freqs, lamb = self.SED(f1=freq_min, f2=freq_max)
        norm_sed = self.normalize_SED(sed, freqs, S_star_ref)

        table_thin = np.array(
            [
                scipy.integrate.simpson(
                    y=self._photo_thin_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                    even="simpson",
                )
                for t in tau
            ]
        )
        table_thick = np.array(
            [
                scipy.integrate.simpson(
                    y=self._photo_thick_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                    even="simpson",
                )
                for t in tau
            ]
        )

        # tables must have shapes: (num taus, num freq) due to the C++ order
        return table_thin.T, table_thick.T

    def make_heat_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]:
        sed, freqs, lamb = self.SED(freq_min, freq_max)
        norm_sed = self.normalize_SED(sed, lamb, S_star_ref)

        table_thin = np.array(
            [
                scipy.integrate.simpson(
                    y=self._heat_thin_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                )
                for t in tau
            ]
        )
        table_thick = np.array(
            [
                scipy.integrate.simpson(
                    y=self._heat_thick_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                )
                for t in tau
            ]
        )

        # tables must have shapes: (num taus, num freq) due to the C++ order
        return table_thin.T, table_thick.T


class BlackBodySource_Multifreq(BlackBodyBase):
    """A point source emitting a Black-body spectrum"""

    def __init__(self, temp: float, grey: bool) -> None:
        self.temp = temp
        self.grey = grey
        # self.freq0 = freq0
        # self.pl_index = pl_index
        self.R_star = 1.0
        self.freq0_HI = (13.598 * u.eV / cst.h).to("Hz").value
        self.freq0_HeI = (24.587 * u.eV / cst.h).to("Hz").value
        self.freq0_HeII = (54.416 * u.eV / cst.h).to("Hz").value

        self.freqs_tab, self.pl_index_HI, self.pl_index_HeI, self.pl_index_HeII = (
            np.loadtxt(
                pc2r.__path__[0] + "/tables/multifreq/Verner1996_spectidx.txt",
                unpack=True,
            )
        )

    def SED(self, freq: float) -> float:
        if freq * h_over_k / self.temp >= 700.0:
            return 0.0
        return (
            4
            * np.pi
            * self.R_star**2
            * two_pi_over_c_square
            * freq**2
            / (np.exp(freq * h_over_k / self.temp) - 1.0)
        )

    def integrate_SED(self, f1: float, f2: float) -> float:
        res, *_ = quad(self.SED, f1, f2)
        return res

    def normalize_SED(self, f1: float, f2: float, S_star_ref: float) -> None:
        S_unscaled = self.integrate_SED(f1, f2)
        S_scaling = S_star_ref / S_unscaled
        self.R_star = np.sqrt(S_scaling) * self.R_star

    def cross_section_freq_dependence(self, freq: float) -> float:
        if self.grey:
            return 1.0

        # MB: use the power-low index of the higher frequency bin (private conversation with Garrelt, Ilian and Sambit), i.e.: use the predominat cross section
        # Not sure if this is correct: see cross-section fit of Verner+ (1996). See Equation 1 and parameters in Table 1.
        if freq < self.freq0_HeI:
            pl_index = np.interp(x=freq, xp=self.freqs_tab, fp=self.pl_index_HI)
            freq0 = self.freq0_HI
        elif freq < self.freq0_HeII and freq >= self.freq0_HeI:
            pl_index = np.interp(x=freq, xp=self.freqs_tab, fp=self.pl_index_HeI)
            freq0 = self.freq0_HeI
        elif freq >= self.freq0_HeII:
            pl_index = np.interp(x=freq, xp=self.freqs_tab, fp=self.pl_index_HeII)
            freq0 = self.freq0_HeII
        return (freq / freq0) ** (-pl_index)

    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        itg = self.SED(freq) * np.exp(-tau * self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _photo_thin_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        itg = (
            self.SED(freq)
            * self.cross_section_freq_dependence(freq)
            * np.exp(-tau * self.cross_section_freq_dependence(freq))
        )
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _heat_thick_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        photo_thick = self._photo_thick_integrand_vec(freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thick

    def _heat_thin_integrand_vec(self, freq: float, tau: FloatArray) -> FloatArray:
        photo_thin = self._photo_thin_integrand_vec(freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thin

    def make_photo_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]:
        self.normalize_SED(freq_min, freq_max, S_star_ref)

        integrand_thin = partial(self._photo_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._photo_thick_integrand_vec, tau=tau)

        # limit the frequency integration based on the provided limit
        # assert freq_min >= self.freqs_tab.min(), "Minimum frequency (freq_min = %.3e Hz) is below value in table %.3e Hz" %(freq_min, self.freqs_tab.min())
        # assert freq_max <= self.freqs_tab.max(), "Maximum frequency (freq_max = %.3e Hz) exceed value in table %.3e Hz" %(freq_max, self.freqs_tab.max())

        # freqs = self.freqs_tab[(self.freqs_tab >= freq_min) * (self.freqs_tab <= freq_max)]
        # freqs = np.linspace(self.freqs_tab.min(), self.freqs_tab.max(), 100)    # TODO: need to be carefull as this can lead to error if the sub-bin is not mentioned in the raytracing
        freqs = self.freqs_tab

        # empty tables
        table_thin = np.zeros((tau.size, freqs.size))
        table_thick = np.zeros((tau.size, freqs.size))

        for i_f, (f_min, f_max) in enumerate(zip(freqs[:-1], freqs[1:])):
            table_thin[:, i_f] = quad_vec(integrand_thin, f_min, f_max, epsrel=1e-12)[0]
            table_thick[:, i_f] = quad_vec(integrand_thick, f_min, f_max, epsrel=1e-12)[
                0
            ]

        # tables must have shapes: (num taus, num freq) due to the C++ order
        return table_thin.T, table_thick.T

    def make_heat_table(
        self, tau: FloatArray, freq_min: float, freq_max: float, S_star_ref: float
    ) -> tuple[FloatArray, FloatArray]:
        self.normalize_SED(freq_min, freq_max, S_star_ref)

        integrand_thin = partial(self._heat_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._heat_thick_integrand_vec, tau=tau)

        # limit the frequency integration based on the provided limit
        # assert freq_min >= self.freqs_tab.min(), "Minimum frequency (freq_min = %.3e Hz) is below value in table %.3e Hz" %(freq_min, self.freqs_tab.min())
        # assert freq_max <= self.freqs_tab.max(), "Maximum frequency (freq_max = %.3e Hz) exceed value in table %.3e Hz" %(freq_max, self.freqs_tab.max())

        # freqs = self.freqs_tab[(self.freqs_tab >= freq_min) * (self.freqs_tab <= freq_max)]
        # freqs = np.linspace(self.freqs_tab.min(), self.freqs_tab.max(), 100)    # TODO: need to be carefull as this can lead to error if the sub-bin is not mentioned in the raytracing
        freqs = self.freqs_tab

        # empty tables
        table_thin = np.zeros((tau.size, freqs.size))
        table_thick = np.zeros((tau.size, freqs.size))

        for i_f, (f_min, f_max) in enumerate(zip(freqs[:-1], freqs[1:])):
            table_thin[:, i_f] = quad_vec(integrand_thin, f_min, f_max, epsrel=1e-12)[0]
            table_thick[:, i_f] = quad_vec(integrand_thick, f_min, f_max, epsrel=1e-12)[
                0
            ]

        # tables must have shapes: (num taus, num freq) due to the C++ order
        return table_thin.T, table_thick.T
