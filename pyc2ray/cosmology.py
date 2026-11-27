from functools import cached_property
from typing import ClassVar, TypeAlias

import astropy.constants as cst
import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FlatLambdaCDM, Parameter
from astropy.cosmology._src.core import dataclass_decorator

__all__ = ["Cosmology21cm"]

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _validate_fraction(cosmology, param: Parameter, value) -> float:
    """Validate that the value is a float in [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{param.name} must be in [0, 1], got {value}.")
    return value


@dataclass_decorator
class Cosmology21cm(FlatLambdaCDM):
    """Cosmology class for PyC2Ray.
    This class inherits  from astropy.cosmology.FlatLambdaCDM and it is used
    to compute a series of quantities related to 21-cm cosmology.
    """

    # Frequency of the 21-cm line
    nu0: ClassVar[u.Quantity] = 1.420405751768e9 * u.Hz

    # Wavelength of the 21-cm line
    lambda0: ClassVar[u.Quantity] = (cst.c / nu0).to(u.cm)

    # Temperature of the 21-cm line
    Tstar: ClassVar[u.Quantity] = (cst.h * nu0 / cst.k_B).to(u.K)

    # Einstein A coefficient for the 21-cm line
    A10: ClassVar[u.Quantity] = 2.85e-15 / u.s

    abu_h: Parameter = Parameter(
        default=0.926,
        doc="Primordial hydrogen number abundance.",
        fvalidate=_validate_fraction,
    )
    abu_he: Parameter = Parameter(
        default=0.074,
        doc="Primordial helium number abundance.",
        fvalidate=_validate_fraction,
    )
    abu_he_mass: Parameter = Parameter(
        default=0.2486,
        doc="Primordial helium mass fraction.",
        fvalidate=_validate_fraction,
    )

    @cached_property
    def abu_h_mass(self) -> float:
        """Primordial hydrogen mass fraction."""
        return 1.0 - self.abu_he_mass

    @cached_property
    def mean_molecular(self) -> float:
        """Mean molecular weight."""
        return self.abu_h + 4.0 * self.abu_he

    @cached_property
    def fHc(self) -> float:
        """Cosmic hydrogen mass fraction."""
        return self.abu_h_mass * self.Ob0 / self.Om0

    @cached_property
    def nH(self) -> u.Quantity:
        """The hydrogen number density from cosmology."""
        return (
            self.abu_h_mass
            * self.Ob0
            * self.critical_density0
            / (self.mean_molecular * cst.m_p)
        ).to(u.cm**-3)

    @cached_property
    def meanT(self) -> u.Quantity:
        return (
            3
            * self.lambda0**3
            * self.A10
            * self.Tstar
            * self.nH
            / (32.0 * np.pi * self.H0)
        ).to(u.mK)

    def mean_nH(self, z: float) -> u.Quantity:
        """Return the mean neutral hydrogen number density from cosmology at redshift z."""
        return self.nH * (1 + z) ** 3

    def z2nu(self, z: float) -> u.Quantity:
        """Return the 21-cm frequency in MHz at the given redshift."""
        return self.nu0.to(u.MHz) / (1 + z)

    def dTb(
        self,
        z: float,
        ndens: FloatArray | u.Quantity | None = None,
        xHII: FloatArray | u.Quantity | None = None,
        Ts: float | u.Quantity | None = None,
        flag_he: bool = False,
    ) -> u.Quantity:
        """Computes the differential brightness temperature in mK given the redshift,
        volume, and optionally the baryon density, neutral hydrogen fraction, neutral
        hydrogen mass, and spin temperature.

        Parameters:
        ------------
        z: The redshift at which to compute the differential brightness temperature.
        ndens: The baryon number density of the region. If not provided, it will be computed from the cosmology.
        xHII: The ionized hydrogen fraction of the region. If not provided, it will be assumed to be 0.
        Ts: The spin temperature of the region. If not provided, it will be assumed to be much greater than the CMB temperature.
        flag_he: a correction due to partially ionized cells when Helium is included will be applied.
        Returns:
        --------
        float
            The differential brightness temperature in mK.
        """
        # TODO: Get this flag from the parameters file.
        if flag_he:
            # TODO: correction due to partially ionised cells when Helium is included
            raise NotImplementedError(
                "Correction due to partially ionised cells not implemented yet."
            )

        if ndens is None:
            ndens = np.atleast_1d(self.mean_nH(z))

        xh_ion = u.Quantity(
            np.zeros(ndens.shape) if xHII is None else xHII,
            u.dimensionless_unscaled,
        )
        if xh_ion.shape != ndens.shape:
            raise ValueError(
                "xHII and ndens must have the same shape. "
                f"Got xHII.shape={xh_ion.shape} and ndens.shape={ndens.shape}."
            )

        # differential brightness temperature
        dT = (
            self.meanT
            * (1 - xh_ion)
            * (ndens / ndens.mean())
            / self.efunc(z)
            * (1 + z) ** 2
        )

        # TODO: a simliar correction is needed to be applied to the spin temperature calculation
        if Ts is not None:
            dT *= 1 - self.Tcmb(z) / u.Quantity(Ts, u.K)

        return dT.to(u.mK)
