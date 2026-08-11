from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import astropy.constants as cst
import astropy.units as u
import numpy as np
import numpy.typing as npt
from scipy import integrate

import pyc2ray.constants as c
from pyc2ray.radiation.radiation_tables import RadiationTables

__all__ = [
    "BlackBodyBase",
    "BlackBodySource",
    "BlackBodySource_Multifreq",
    "YggdrasilModel",
]

BlackBodyType = TypeVar("BlackBodyType", bound="BlackBodyBase")

FloatArray = npt.NDArray[np.float64]


class BlackBodyBase(ABC):
    @abstractmethod
    def make_photo_tables(self, tau: FloatArray) -> tuple[FloatArray, FloatArray]: ...

    @abstractmethod
    def make_heat_tables(self, tau: FloatArray) -> tuple[FloatArray, FloatArray]: ...


class BlackBodySource(BlackBodyBase):
    """A point source emitting a Black-body spectrum.
    We distinguish between optically thin and thick cells to better approximate exponential calculations.
    See radiation_tables.f90:345
    """

    INTEGRATION_TOLERANCE = 1e-12

    def __init__(
        self,
        temp: float,
        freq_range: tuple[float, float],
        ion_freq: float,
        pl_index: float,
        grey: bool = False,
        S_start_freq: float = 1e48,
    ) -> None:
        self.temp = temp
        self.grey = grey
        self.ion_freq = ion_freq
        self.pl_index = pl_index
        self.S_star_ref = S_start_freq

        # Integration parameters.
        self.quad_kws = dict(
            a=freq_range[0], b=freq_range[1], epsrel=self.INTEGRATION_TOLERANCE
        )

        self.R_star2 = self._normalize_SED()

    def SED(self, freq: float) -> float:
        """Spectral energy distribution."""
        nu = freq * c.h / c.k_B / self.temp
        return 2.0 * c.pi / c.c**2 * freq**2 / (np.exp(nu) - 1.0)

    def _normalize_SED(self, **kwargs) -> float:
        """Integrate the SED to estimate the radius of the source."""
        kwargs = kwargs or self.quad_kws
        with np.errstate(over="ignore"):
            S_unscaled = 4.0 * c.pi * integrate.quad(self.SED, **kwargs)[0]
        return self.S_star_ref / S_unscaled

    def integrand_thick(
        self,
        tau: FloatArray,
        freq: float,
        ion_freq: float,
        pl_index: float,
    ) -> FloatArray:
        """Integrand term for the photoionization rate in optically thick cells.
        The frequency dependence is included via the term `dep`.
        Parameters
        ----------
        tau :
            Optical depth array at which to evaluate the integrand.
        freq :
            The current frequency at which to evaluate the integrand.
        ion_freq :
            Ionization frequency for the species being considered.
        pl_index :
            Power-law index for the frequency dependence of the cross-section.
        Returns
        -------
        integrand: array
            The integrand evaluated at the given frequency and optical depth.
        """
        dep = (freq / ion_freq) ** (-pl_index) if not self.grey else 1.0
        return 4.0 * np.pi * self.R_star2 * self.SED(freq) * np.exp(-tau * dep)

    def integrand_thin(
        self,
        tau: FloatArray,
        freq: float,
        ion_freq: float,
        pl_index: float,
    ) -> FloatArray:
        """Integrand term for the photoionization rate in optically thin cells.
        The frequency dependence is included via the term `dep`.
        Parameters
        ----------
        tau :
            Optical depth array at which to evaluate the integrand.
        freq :
            The current frequency at which to evaluate the integrand.
        ion_freq :
            Ionization frequency for the species being considered.
        pl_index :
            Power-law index for the frequency dependence of the cross-section.
        Returns
        -------
        integrand: array
            The integrand evaluated at the given frequency and optical depth.
        """
        dep = (freq / ion_freq) ** (-pl_index) if not self.grey else 1.0
        return 4.0 * np.pi * self.R_star2 * self.SED(freq) * dep * np.exp(-tau * dep)

    def make_photo_integrand(self, integrand: Callable, tau: FloatArray) -> Callable:
        """Create the integrand term by capturing the optical depth array."""

        def func(freq: float) -> FloatArray:
            return integrand(tau, freq, self.ion_freq, self.pl_index)

        return func

    def make_heat_integrand(self, integrand: Callable, tau: FloatArray) -> Callable:
        """Create the integrand term by capturing the optical depth array."""

        def func(freq: float) -> FloatArray:
            if freq < self.ion_freq:
                return np.zeros_like(tau)
            fact = c.h * (freq - self.ion_freq)
            return fact * integrand(tau, freq, self.ion_freq, self.pl_index)

        return func

    def make_photo_tables(self, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Create tables for photoionization rates. The tables are 1D arrays with shape (tau,)"""
        thin = self.make_photo_integrand(self.integrand_thin, tau)
        table_thin = integrate.quad_vec(thin, **self.quad_kws)[0]

        with np.errstate(over="ignore"):
            thick = self.make_photo_integrand(self.integrand_thick, tau)
            table_thick = integrate.quad_vec(thick, **self.quad_kws)[0]

        return table_thin, table_thick

    def make_heat_tables(self, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Create tables for photoheating rates. The tables are 1D arrays with shape (tau,)"""
        with np.errstate(over="ignore"):
            thin = self.make_heat_integrand(self.integrand_thin, tau)
            table_thin = integrate.quad_vec(thin, **self.quad_kws)[0]

            thick = self.make_heat_integrand(self.integrand_thick, tau)
            table_thick = integrate.quad_vec(thick, **self.quad_kws)[0]

        return table_thin, table_thick


class BlackBodySource_Multifreq(BlackBodySource):
    """A point source composed of HI, HeI and HeII emitting a Black-body spectrum"""

    def __init__(
        self,
        temp: float,
        grey: bool = False,
        S_star_ref: float = 1e48,
    ) -> None:
        self.temp = temp
        self.grey = grey
        self.S_star_ref = S_star_ref

        rt = RadiationTables()
        self.freq_min, self.freq_max = rt.freqs
        self.pl_index_HI, self.pl_index_HeI, self.pl_index_HeII = rt.powerlaw_indices

        self.ion_freq_HI = rt.ion_freq_HI
        self.ion_freq_HeI = rt.ion_freq_HeI
        self.ion_freq_HeII = rt.ion_freq_HeII

        self.quad_kws = dict(epsrel=self.INTEGRATION_TOLERANCE)
        self.R_star2 = self._normalize_SED(
            a=self.freq_min[0], b=self.freq_max[-1], **self.quad_kws
        )

    def make_photo_integrand(
        self,
        integrand: Callable,
        tau: FloatArray,
        freq_min: float = 0.0,
        pl_index: float = 0.0,
    ) -> Callable:
        """Create the integrand term by capturing the optical depth array and other parameters."""

        def func(freq: float) -> FloatArray:
            return integrand(tau, freq, freq_min, pl_index)

        return func

    def make_heat_integrand(
        self,
        integrand: Callable,
        tau: FloatArray,
        freq_min: float = 0.0,
        ion_freq: float = 0.0,
        pl_index: float = 0.0,
    ) -> Callable:
        """Create the integrand term by capturing the optical depth array and other parameters."""

        def func(freq: float) -> FloatArray:
            if freq < ion_freq:
                return np.zeros_like(tau)
            fact = c.h * (freq - ion_freq)
            return fact * integrand(tau, freq, freq_min, pl_index)

        return func

    def _select_powerlaw_index(self, freq: float) -> np.ndarray:
        """Find the correct power-law index"""
        if freq < self.ion_freq_HeI:
            return self.pl_index_HI
        elif freq < self.ion_freq_HeII:
            return self.pl_index_HeI
        return self.pl_index_HeII

    def make_photo_tables(self, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Create tables for photoionization rates. The tables are 2D arrays with dimensions (tau, freq)"""
        tables = np.empty((2, len(self.freq_min), len(tau)), dtype=np.float64)

        for i, (fmin, fmax) in enumerate(zip(self.freq_min, self.freq_max)):
            pl = self._select_powerlaw_index(fmin)[i]

            with np.errstate(over="ignore"):
                thin = self.make_photo_integrand(self.integrand_thin, tau, fmin, pl)
                tables[0, i] = integrate.quad_vec(thin, fmin, fmax, **self.quad_kws)[0]

                thick = self.make_photo_integrand(self.integrand_thick, tau, fmin, pl)
                tables[1, i] = integrate.quad_vec(thick, fmin, fmax, **self.quad_kws)[0]

        return tables[0], tables[1]

    def make_heat_tables(self, tau: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Create tables for heating rates. The tables are 3D arrays with dimensions (tau, freq, ion_species)"""
        tables = np.empty((2, len(self.freq_min), 3, len(tau)), dtype=np.float64)

        for i, (fmin, fmax) in enumerate(zip(self.freq_min, self.freq_max)):
            pl = self._select_powerlaw_index(fmin)[i]

            for j, ion_freq in enumerate(
                (self.ion_freq_HI, self.ion_freq_HeI, self.ion_freq_HeII)
            ):
                with np.errstate(over="ignore"):
                    thin = self.make_heat_integrand(
                        self.integrand_thin, tau, fmin, ion_freq, pl
                    )
                    tables[0, i, j] = integrate.quad_vec(
                        thin, fmin, fmax, **self.quad_kws
                    )[0]

                    thick = self.make_heat_integrand(
                        self.integrand_thick, tau, fmin, ion_freq, pl
                    )
                    tables[1, i, j] = integrate.quad_vec(
                        thick, fmin, fmax, **self.quad_kws
                    )[0]

        return tables[0], tables[1]


class YggdrasilModel(BlackBodyBase):
    """Use Yggdrasil model for SED"""

    def __init__(
        self,
        tabname: str,
        grey: bool,
        freq_range: tuple[float, float],
        ion_freq: float,
        pl_index: float,
        S_star_ref: float = 1e48,
    ) -> None:
        self.grey = grey
        self.ion_freq = ion_freq
        self.tabname = tabname
        self.pl_index = pl_index
        self.f_min, self.f_max = freq_range
        self.S_star_ref = S_star_ref

    def SED(self, f1: float, f2: float) -> tuple[float, FloatArray, FloatArray]:
        """This was used for debugging. Can be usefull in the future(?)

        freqs = np.linspace(f2, f1, 10) * u.Hz
        lamb = (cst.c  / freqs).to('AA')
        R_star, temp = 1*u.Rsun, 5e4*u.K

        #ampl = 8*c.pi**2 * R_star**2 *cst.h * cst.c**2 / lamb**5
        #sed = ampl / (np.exp((cst.h*cst.c/lamb/cst.k_B/temp).cgs)-1.0)
        ampl = 8*c.pi**2 * R_star**2 *cst.h * freqs**5/cst.c**3
        sed = ampl / (np.exp(freqs*cst.h/cst.k_B/temp)-1.0)

        return sed.to('erg / s / AA').value, freqs.value, lamb.value
        """
        lamb, flux = np.loadtxt(
            self.tabname, unpack=True
        )  # wavelenght in (Angstrom), Flux in (erg/s/AA)
        freqs = (cst.c / (lamb * u.AA)).to("Hz").value

        if (freqs.min() == freqs[-1]) and (freqs.max() == freqs[0]):
            # this is for the discrete integral integrate.simpson, which require increasing value for the x-axis
            lamb = lamb[::-1]
            freqs = freqs[::-1]
            flux = flux[::-1]

        int_range = (freqs >= f1) * (freqs <= f2)
        sed = flux[int_range]
        return sed, freqs[int_range], lamb[int_range]

    def integrate_SED(self, sed: float, freq: FloatArray) -> float:
        assert freq.min() == freq[0]
        assert freq.max() == freq[-1]

        return integrate.simpson(x=freq, y=sed)

    def normalize_SED(self, sed: float, freq: FloatArray, S_star_ref: float) -> float:
        S_unscaled = self.integrate_SED(sed, freq)
        # MB: in C2Ray this was: self.R_star = np.sqrt(S_scaling) * self.R_star.
        # Here we define the SED with the proper units so we do not need to squareroot
        # (as we do not multiply to R_star) and instead multiply directly to the SED.
        S_scaling = S_star_ref / S_unscaled
        return sed * S_scaling

    def cross_section_freq_dependence(self, freq: FloatArray) -> FloatArray:
        if self.grey:
            return np.ones_like(freq)
        return (freq / self.ion_freq) ** (-self.pl_index)

    # C2Ray distinguishes between optically thin and thick cells,
    # and calculates the rates differently for those two cases.
    # See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        dep = self.cross_section_freq_dependence(freq)
        itg = sed * np.exp(-tau * dep)
        return np.where(tau * dep < 700.0, itg, 0.0)

    def _photo_thin_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        dep = self.cross_section_freq_dependence(freq)
        itg = sed * dep * np.exp(-tau * dep)
        return np.where(tau * dep < 700.0, itg, 0.0)

    def _heat_thick_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        photo_thick = self._photo_thick_integrand_vec(sed, freq, tau)
        return c.h * (freq - self.ion_freq) * photo_thick

    def _heat_thin_integrand_vec(
        self, sed: float, freq: FloatArray, tau: FloatArray
    ) -> FloatArray:
        photo_thin = self._photo_thin_integrand_vec(sed, freq, tau)
        return c.h * (freq - self.ion_freq) * photo_thin

    def make_photo_tables(
        self,
        tau: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        sed, freqs, lamb = self.SED(f1=self.f_min, f2=self.f_max)
        norm_sed = self.normalize_SED(sed, freqs, self.S_star_ref)

        table_thin = np.array(
            [
                integrate.simpson(
                    y=self._photo_thin_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                    even="simpson",
                )
                for t in tau
            ]
        )
        table_thick = np.array(
            [
                integrate.simpson(
                    y=self._photo_thick_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                    even="simpson",
                )
                for t in tau
            ]
        )

        # tables must have shapes: (num taus, num freq) due to the C++ order
        return table_thin.T, table_thick.T

    def make_heat_tables(
        self,
        tau: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        sed, freqs, lamb = self.SED(self.f_min, self.f_max)
        norm_sed = self.normalize_SED(sed, lamb, self.S_star_ref)

        table_thin = np.array(
            [
                integrate.simpson(
                    y=self._heat_thin_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                )
                for t in tau
            ]
        )
        table_thick = np.array(
            [
                integrate.simpson(
                    y=self._heat_thick_integrand_vec(sed=norm_sed, freq=freqs, tau=t),
                    x=freqs,
                )
                for t in tau
            ]
        )

        # tables must have shapes: (num taus, num freq) due to the C++ order
        return table_thin.T, table_thick.T
