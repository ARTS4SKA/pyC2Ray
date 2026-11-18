from functools import cached_property

import astropy.constants as cst
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

import pyc2ray.constants as c

__all__ = ["Cosmology21cm"]


class Cosmology21cm(FlatLambdaCDM):
    """Cosmology class for PyC2Ray
    This class is inherited from astropy.cosmology.FlatLambdaCDM and it is used to compute a series of quantities related to 21-cm cosmology.
    z2nu:
        Converts redshift to frequency in MHz
    dTb:
        Computes the differential brightness temperature in mK given the redshift, volume, and optionally the baryon density, neutral hydrogen fraction, neutral hydrogen mass, and spin temperature.
    """

    def __init__(self, abu_h, abu_he, abu_he_mass, *args, **kwargs):
        """Initializes the Cosmology21cm class with a given cosmology and primordial helium abundance.

        Parameters:
        ------------
        cosmo: astropy.cosmology.FlatLambdaCDM
            The cosmology to use for calculations.
        abu_h: float, optional
            The primordial hydrogen abundance.
        abu_he: float, optional
            The primordial helium abundance.
        abu_he_mass: float, optional
            The primordial helium mass fraction.
        """
        super().__init__(*args, **kwargs)

        # primordial nucleosynthesis quantities
        self.abu_he = abu_he
        self.abu_h = abu_h
        self.abu_he_mass = abu_he_mass
        self.abu_h_mass = 1.0 - self.abu_he_mass
        self.mean_molecular = self.abu_h + 4.0 * self.abu_he

        # cosmic hydrogen mass fraction
        self.fHc = self.abu_h_mass * self.Ob0 / self.Om0

        # 21-cm astrophysics quantities
        self.nu0 = c.nu0 * u.MHz
        self.lamb0 = c.lamb0 * u.cm

        # differential brightness quantities
        self.A10 = c.A10 / u.s

    @cached_property
    def nH(self):
        return (
            self.abu_h_mass
            * self.Ob0
            * self.critical_density0
            / (self.mean_molecular * cst.m_p)
        ).cgs

    @cached_property
    def Tstar(self):
        return (cst.h * self.nu0 / cst.k_B).to(u.K)

    @cached_property
    def meanT(self):
        return (
            3
            * self.lamb0**3
            * self.A10
            * self.Tstar
            / (32.0 * np.pi)
            * self.nH
            / self.H0
        ).to(u.mK)

    def mean_nH(self, z: float) -> float:
        """Computes the mean neutral hydrogen number density in cm^-3 from cosmology.
        Parameters:
        ------------
        z: float
            The redshift at which to compute the neutral hydrogen number density.
        Returns:
        --------
        float
            The neutral hydrogen number density in cm^-3."""
        return self.nH.cgs.value * (1 + z) ** 3

    def z2nu(self, z: float) -> float:
        """Converts redshift to frequency in MHz.
        Parameters:
        ------------
        z: float
            The redshift to convert.
        Returns:
        --------
        float
            The frequency in MHz corresponding to the given redshift."""
        return self.nu0.to(u.MHz).value / (1 + z)

    def dTb(
        self,
        z: float,
        ndens=None,
        xHII=None,
        Ts=None,
        flag_he: bool = False,
    ) -> float:
        """Computes the differential brightness temperature in mK given the redshift, volume, and optionally the baryon density, neutral hydrogen fraction, neutral hydrogen mass, and spin temperature.
        Parameters:
        ------------
        z: float
            The redshift at which to compute the differential brightness temperature.
        ndens: float, optional
            The baryon number density of the region. If not provided, it will be computed from the cosmology.
        xHII: float, optional
            The ionized hydrogen fraction of the region. If not provided, it will be assumed to be 0.
        Ts: float, optional
            The spin temperature of the region. If not provided, it will be assumed to be much greater than the CMB temperature.
        flag_he: bool, optional
            If True, a correction due to partially ionized cells when Helium is included will be applied. If False, no correction will be applied. Default is False.
            TODO: we will require to read the same flag from the parameters file and pass it to this function. This will be done in a future update.
        Returns:
        --------
        float
            The differential brightness temperature in mK.
        """
        try:
            ndens = ndens.to(u.cm**-3)
        except AttributeError:
            ndens *= u.cm**-3

        # differential brightness temperature in mK
        dT = (
            self.meanT
            * (1 - xHII)
            * ndens
            / ndens.mean()
            / self.efunc(z)
            * (1 + z) ** 2
        )

        if flag_he:
            # TODO: correction due to partially ionised cells when Helium is included
            raise NotImplementedError(
                "Correction due to partially ionised cells not implemented yet."
            )

        # TODO: a simliar correction is needed to be applied to the spin temperature calculation
        if Ts != None:
            dT *= 1 - self.Tcmb(z) / Ts

        return dT.to("mK").value
