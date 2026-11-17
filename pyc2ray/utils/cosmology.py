import astropy.constants as cst
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# constants for 21-cm physics
nu0 = 1420.405751768 * u.MHz
lamb0 = (cst.c / nu0).cgs


class PyC2Ray_Cosmology(FlatLambdaCDM):
    """Cosmology class for PyC2Ray
    This class is inherited from astropy.cosmology.FlatLambdaCDM and it is used to compute a series of quantities related to 21-cm cosmology.
    z2nu:
        Converts redshift to frequency in MHz
    dTb:
        Computes the differential brightness temperature in mK given the redshift, volume, and optionally the baryon density, neutral hydrogen fraction, neutral hydrogen mass, and spin temperature.
    """

    def __init__(
        self,
        cosmo=None,
        abu_he=0.074,
        abu_he_mass=0.2486,
        *,
        H0=None,
        Om0=None,
        Ob0=None,
        Tcmb0=None,
        Neff=None,
        m_nu=None,
    ):
        """Initializes the PyC2Ray_Cosmology class with a given cosmology and primordial helium abundance.

        Parameters:
        ------------
        cosmo: astropy.cosmology.FlatLambdaCDM
            The cosmology to use for calculations.
        abu_he: float, optional
            The primordial helium abundance.
        abu_he_mass: float, optional
            The primordial helium mass fraction.
        """
        if cosmo is None:
            # if not all the required parameters are provided, raise an error
            missing = [
                name
                for name, value in (("H0", H0), ("Om0", Om0), ("Ob0", Ob0))
                if value is None
            ]
            if missing:
                raise TypeError(
                    f"Provide either cosmo or the required parameters: {', '.join(missing)}"
                )

            params = {"H0": H0, "Om0": Om0, "Ob0": Ob0}
            params.update(
                {
                    name: value
                    for name, value in (
                        ("Tcmb0", Tcmb0),
                        ("Neff", Neff),
                        ("m_nu", m_nu),
                    )
                    if value is not None
                }
            )
            cosmo = FlatLambdaCDM(**params)
        elif any(value is not None for value in (H0, Om0, Ob0, Tcmb0, Neff, m_nu)):
            raise TypeError("Pass either cosmo or cosmological parameters, not both")

        super().__init__(
            H0=cosmo.H0,
            Om0=cosmo.Om0,
            Ob0=cosmo.Ob0,
            Tcmb0=cosmo.Tcmb0,
            Neff=cosmo.Neff,
            m_nu=cosmo.m_nu,
        )
        self.cosmo = cosmo

        # primordial nucleosynthesis quantities
        self.abu_he = abu_he
        self.abu_h = 1.0 - self.abu_he
        self.abu_he_mass = abu_he_mass
        self.abu_h_mass = 1.0 - self.abu_he_mass
        self.mean_molecular = self.abu_h + 4.0 * self.abu_he

        # cosmic hydrogen mass fraction
        self.fHc = self.abu_h_mass * self.cosmo.Ob0 / self.cosmo.Om0

        # 21-cm astrophysics quantities
        self.nu0 = nu0
        self.lamb0 = lamb0
        self.nH = (
            self.abu_h_mass
            * cosmo.Ob0
            * cosmo.critical_density0
            / (self.mean_molecular * cst.m_p)
        ).cgs

        # differential brightness quantities
        self.A10 = 2.85e-15 / u.s
        self.Tstar = (cst.h * self.nu0 / cst.k_B).to(u.K)
        self.meanT = (
            3
            * self.lamb0**3
            * self.A10
            * self.Tstar
            / (32.0 * np.pi)
            * self.nH
            / cosmo.H0
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
        return self.nu0.to(u.MHz) / (1 + z)

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
            / self.cosmo.efunc(z)
            * (1 + z) ** 2
        )

        if flag_he:
            # TODO: correction due to partially ionised cells when Helium is included
            raise NotImplementedError(
                "Correction due to partially ionised cells not implemented yet."
            )

        # TODO: a simliar correction is needed to be applied to the spin temperature calculation
        if Ts != None:
            dT *= 1 - self.cosmo.Tcmb(z) / Ts

        return dT.to("mK").value
