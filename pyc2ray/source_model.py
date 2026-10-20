from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad_vec
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_dd

from pyc2ray.parameters import (
    ConstantFescParameters,
    DplSourceParameters,
    ExpAccretionParameters,
    FescParameters,
    FgammaSourceParameters,
    FstarParameters,
    MuvSourceParameters,
    PowerFescParameters,
    PowerObsFescParameters,
    ThesanFescParameters,
)

FloatLike = float | npt.NDArray[np.float64]

# Conversion Factors.
# When doing direct comparisons with C2Ray, the difference between astropy.constants and the C2Ray values may be visible, thus we use the same exact value for the constants. This can be changed to the astropy values once consistency between the two codes has been established
# pc = 3.086e18           #(1*u.pc).to('cm').value            # C2Ray value: 3.086e18
# YEAR = 3.15576E+07      #(1*u.yr).to('s').value           # C2Ray value: 3.15576E+07
# ev2fr = 0.241838e15                     # eV to Frequency (Hz)
# ev2k = 1.0/8.617e-05                    # eV to Kelvin
# kpc = 1e3*pc                            # kiloparsec in cm
# Mpc = 1e6*pc                            # megaparsec in cm
# msun2g = 1.98892e33 #(1*u.Msun).to('g').value       # solar mass to grams
# m_p = 1.672661e-24


class StellarToHaloRelation:
    """Modelling the mass relation between dark matter halo and the residing stars/galaxies."""

    def __init__(
        self,
        params: FstarParameters,
        cosmo: FlatLambdaCDM,
        rng: np.random.Generator | None = None,
    ):
        self.cosmo = cosmo
        self.rng = rng if rng is not None else np.random.default_rng()
        self.model = params.MODEL

        self.get: Callable[..., FloatLike]
        if isinstance(params, FgammaSourceParameters):
            self.f0 = params.f0
            # TODO: there is something wrong with this, The photoionization gets super high like 1e-5?????? to check
            self.get = (
                lambda Mhalo: self.cosmo.Ob0 / self.cosmo.Om0 * Mhalo * self.f0  # type: ignore
            )
        elif isinstance(params, (DplSourceParameters, MuvSourceParameters)):
            self.f0 = params.f0
            self.Mt = params.Mt
            self.Mp = params.Mp
            self.g1 = params.g1
            self.g2 = params.g2
            self.g3 = params.g3
            self.g4 = params.g4
            if isinstance(params, DplSourceParameters):
                self.get = self.deterministic
            else:
                self.get = self.fstar_from_Muv
        else:
            raise TypeError(
                f"Selected stellar-to-halo relation model that does not exist : {self.model}"
            )

    def source_lifetime(self, z: float, alpha_h: float) -> float:
        ts = 1.0 / (alpha_h * (1 + z) * self.cosmo.H(z).cgs.value)
        return ts

    def deterministic(self, Mhalo: FloatLike) -> FloatLike:
        fstar_mean = self.stellar_to_halo_fraction(Mhalo)
        return fstar_mean

    def stochastic_Gaussian(self, Mhalo: FloatLike, sigma: FloatLike) -> FloatLike:
        fstar_mean = self.stellar_to_halo_fraction(Mhalo)

        if isinstance(sigma, float):
            fstar_std = np.full_like(Mhalo, sigma)
        else:
            fstar_std = sigma

        fstar = np.clip(
            fstar_mean * (1 + self.rng.normal(0, fstar_std)),
            a_min=0,
            a_max=1,
        )

        return fstar

    def fstar_from_Muv(
        self,
        Mhalo: FloatLike,
        z: float,
        alpha_h: float,
        a_s: float = -0.33334,
        b_s: float = 4.5,
    ) -> FloatLike:
        # source life-time (for accreation mass) in cgs units
        ts = self.source_lifetime(z, alpha_h)

        # mean absolute magnitude
        mean_fstar = self.stellar_to_halo_fraction(Mhalo=Mhalo)
        mean_Muv = self.UV_magnitude(fstar=mean_fstar, mdot=Mhalo / ts)

        # following Gelli+ (2024), Muv scatter is proportional to halo circular velocity: ~M^(-1/3)
        std_Muv = a_s * np.log10(Mhalo) + b_s
        # np.log10(Mhalo**a_s*10**b_s) a*x_b
        # absolute magnitude with scatter
        Muv = self.rng.normal(loc=mean_Muv, scale=std_Muv)

        # calibrated for 1500 Å dust-corrected rest-frame UV luminosity
        M0, k_val = 51.6, 3.64413e-36  # in [Msun/s * Hz / (s erg)]
        fstar = (
            self.cosmo.Om0
            / self.cosmo.Ob0
            * k_val
            / (Mhalo / ts)
            * np.power(10.0, (M0 - Muv) / 2.5)
        )
        return np.clip(fstar, 0.0, 1.0)

    def stellar_to_halo_fraction(self, Mhalo: FloatLike) -> FloatLike:
        """
        A parameterised stellar to halo relation (2011.12308, 2201.02210, 2302.06626).
        """
        # Double power law, motivated by UVLFs
        dpl = (
            2
            * self.cosmo.Ob0
            / self.cosmo.Om0
            * self.f0
            / ((Mhalo / self.Mp) ** self.g1 + (Mhalo / self.Mp) ** self.g2)
        )

        # Suppression at the small-mass end
        S_M = (1 + (self.Mt / Mhalo) ** self.g3) ** self.g4

        fstar = dpl * S_M

        return fstar

    def UV_magnitude(self, fstar: FloatLike, mdot: FloatLike) -> FloatLike:
        # corresponding to AB magnitude system (Oke 1974)
        M0 = 51.6

        # calibrated for 1500 Å dust-corrected rest-frame UV luminosity
        # k_val = 1.15e-28 # in [Msun/yr * Hz / (s erg)]
        k_val = 3.64413e-36  # in [Msun/s * Hz / (s erg)]

        M_UV = M0 - 2.5 * (
            np.log10(fstar)
            + np.log10(self.cosmo.Ob0 / self.cosmo.Om0)
            + np.log10(mdot / k_val)
        )
        return M_UV


class EscapeFraction:
    """Modelling the escape of photons from the stars/galaxies inside dark matter haloes."""

    def __init__(self, params: FescParameters) -> None:
        self.model = params.MODEL

        self.get: Callable[..., FloatLike]
        if isinstance(params, ConstantFescParameters):
            self.f0_esc = params.f0_esc
            self.get = lambda Mhalo: np.full_like(Mhalo, self.f0_esc)
        elif isinstance(params, (PowerFescParameters, PowerObsFescParameters)):
            self.f0_esc = params.f0_esc
            self.Mp_esc = params.Mp_esc
            self.al_esc = params.al_esc
            self.get = self.deterministic
        elif isinstance(params, ThesanFescParameters):
            # path to the tables
            tables_dir = Path(params.tables_dir)

            # get fesc tables and bins for interpolation
            tabs = np.loadtxt(tables_dir / "fesc_thesan_tables.txt")
            self.redshift_tab = np.loadtxt(tables_dir / "redshifts.txt")
            mass_tab = np.loadtxt(tables_dir / "mass_bin.txt")
            self.mass_mid = 0.5 * (mass_tab[1:] + mass_tab[:-1])

            # use a 2D interpolation based on the redshift and mass bins
            self.interp_func = RegularGridInterpolator(
                (self.redshift_tab, self.mass_mid), tabs
            )

            self.get = self.fesc_Thesan
        else:
            raise TypeError(
                f"Selected escaping fraction model that does not exist : {self.model}"
            )

    def deterministic(self, Mhalo) -> FloatLike:
        fesc_mean = self.f0_esc * (Mhalo / self.Mp_esc) ** self.al_esc
        return np.clip(fesc_mean, 0, 1)

    def deterministic_redshift(self, z: float) -> float:
        fesc_mean = self.f0_esc * (1 + z) ** self.al_esc
        return np.clip(fesc_mean, 0, 1)

    def fesc_Thesan(self, Mhalo, z) -> FloatLike:
        if z > self.redshift_tab.max():
            fesc = self.interp_func(
                np.array([np.full_like(Mhalo, self.redshift_tab.max()), Mhalo]).T
            )
        elif z < self.redshift_tab.min():
            fesc = self.interp_func(
                np.array([np.full_like(Mhalo, self.redshift_tab.min()), Mhalo]).T
            )
        else:
            fesc = self.interp_func(np.array([np.full_like(Mhalo, z), Mhalo]).T)

        return np.clip(fesc, 0, 1)


class BurstySFR:
    """Modelling bursty star formation"""

    def __init__(
        self,
        params: ExpAccretionParameters,
        cosmo: FlatLambdaCDM,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = params.bursty_sfr
        self.beta1 = params.beta1
        self.beta2 = params.beta2
        self.tB0 = params.tB0
        self.tQ_frac = params.tQ_frac
        self.z0 = params.z0
        self.t_rnd = params.t_rnd
        self.alpha_h = params.alpha_h
        self.cosmo = cosmo
        self.rng = rng if rng is not None else np.random.default_rng()

        self.t0 = cosmo.age(self.z0).to("Myr").value

        self.get: Callable[[FloatLike, float], npt.NDArray[np.bool_]]
        if self.model == "instant":
            self.get = self.instant_burst_or_quiescent_galaxies
        elif self.model == "integrate":
            raise ValueError(" Sorry, model not yet implemented : %s" % self.model)
        elif self.model == "no":
            raise ValueError(
                " You have selected %s model. You should not call this class or change the variable in the parameter file."
                % self.model
            )
        else:
            raise ValueError(
                " Selected burstiness model that does not exist : %s" % self.model
            )

    def time_burstiness(self, mass: FloatLike, z: float) -> FloatLike:
        M0: FloatLike
        if self.t_rnd:
            # FIXME: M0 used in lhs and rhs, this is a bug
            M0 = 10 ** self.rng.normal(np.log10(M0), self.t_rnd)  # noqa: F821
        else:
            M0 = mass / np.exp(-self.alpha_h * (z - self.z0))

        t = self.cosmo.age(z).to("Myr").value

        # burstiness time [Myr]
        tB = (
            self.tB0
            * (M0 / 1e10) ** self.beta1
            * ((t - self.t0) * self.cosmo.H(z).to("1/Myr").value) ** self.beta2
        )

        return tB

    @np.vectorize
    def _burstiness_timescale(t_age: float, tB: float, tQ: float) -> int:
        """of internal use for the integrated_burst_or_quiescent_galaxies method"""
        i_time = np.floor(t_age / (tB + tQ))

        return int(t_age <= i_time * (tB + tQ) + tB)

    def integrated_burst_or_quiescent_galaxies(
        self, mass: FloatLike, z: float, zi: float, zf: float, cosmo: FlatLambdaCDM
    ) -> FloatLike:
        """This case integrate the burst or quench time withing the time-step.
        It return a factor between 0 and 1 for quenched (value 0) or bursting (value 1).
        In bewteen values indicate that the sources are quencing for a period of time withing the time-step.
        """
        # TODO: It is computationally expensive, for some reason, due to the quad_vec method.... to investiage

        # get burstiness and quencing time
        tB = self.time_burstiness(mass, z)
        tQ = self.tQ_frac * tB

        # get time interval limits
        ti = cosmo.age(zi).to("Myr").value - self.t0
        tf = cosmo.age(zf).to("Myr").value - self.t0

        # get time fraction that the galaxies are on
        integr = partial(self._burstiness_timescale, tB=tB, tQ=tQ)
        timefrac_on = quad_vec(integr, ti, tf)[0] / (tf - ti)

        return timefrac_on

    def instant_burst_or_quiescent_galaxies(
        self, mass: FloatLike, z: float
    ) -> npt.NDArray[np.bool_]:
        """This case is for instanteneous bursting or quenching. Do not account for the time integration. Mask the halo True (bursting) or False (quiescent)."""
        # get burstiness and quencing time
        tB = self.time_burstiness(mass, z)
        tQ = self.tQ_frac * tB
        # tB *= (1-self.tQ_frac)

        # get time at the corresponding redshift
        t_age = self.cosmo.age(z).to("Myr").value - self.t0
        assert t_age.all() > 0.0, (
            "Selected parameter t0 is wrong. The value of z0 is lower then the redshift of the first source file (increase the value z0)."
        )

        # find the index of the burst/quench cycle in which the time-step, t, is inside
        i_time = np.floor(t_age / (tB + tQ))

        # if True then the galaxy is bursting otherwise is quenching
        burst_or_quench = t_age <= i_time * (tB + tQ) + tB

        # print(' A total of %.2f %% of galaxies have bursty star-formation.' %(100*np.count_nonzero(burst_mask)/burst_mask.size))
        return burst_or_quench


class Halo2Grid:
    def __init__(self, box_len, n_grid, method="nearest"):
        self.box_len = box_len
        self.n_grid = n_grid

        self.mpc_to_cm = 3.085677581491367e24  # in cm
        self.Msun_to_g = 1.988409870698051e33  # in gram
        self.pos_grid = None

    def set_halo_pos(self, pos, unit=None):
        if unit.lower() == "cm":
            self.pos_cm_to_grid(pos)
        elif unit.lower() == "mpc":
            self.pos_mpc_to_grid(pos)
        else:
            self.pos_grid = pos

    def set_halo_mass(self, mass, unit=None):
        if unit.lower() == "kg":
            self.mass_Msun = mass * 1000 / self.Msun_to_g
        elif unit.lower() in ["gram", "g"]:
            self.mass_Msun = mass / self.Msun_to_g
        elif unit.lower() == "msun":
            self.mass_Msun = mass
        else:
            print("Unknown mass units")

    def pos_cm_to_grid(self, pos_cm):
        pos_mpc = pos_cm / self.mpc_to_cm
        pos_grid = pos_mpc * self.n_grid / self.box_len
        self.pos_grid = pos_grid
        print("Halo positions converted from cm to grid units")
        return pos_grid

    def pos_mpc_to_grid(self, pos_mpc):
        pos_grid = pos_mpc * self.n_grid / self.box_len
        self.pos_grid = pos_grid
        print("Halo positions converted from Mpc to grid units")
        return pos_grid

    def construct_tree(self, **kwargs):
        pos = kwargs.get("pos", self.pos_grid)
        if pos is None:
            print('Provide the halo positions via parameter "pos".')
            return

        print("Creating a tree...")
        kdtree = cKDTree(pos)
        self.kdtree = kdtree
        print("...done")

    def value_on_grid(self, positions, values, **kwargs):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_dd.html
        statistic = kwargs.get("statistic", "sum")
        bins = kwargs.get("bins", self.n_grid)
        binned_mass, bin_edges, bin_num = binned_statistic_dd(
            positions, values, statistic=statistic, bins=bins
        )
        return binned_mass, bin_edges, bin_num

    def halo_mass_on_grid(self, **kwargs):
        pos = kwargs.get("pos", self.pos_grid)
        if pos is None:
            print('Provide the halo positions via parameter "pos".')
            return None
        mass = kwargs.get("mass", self.pos_grid)
        if mass is None:
            print('Provide the halo masses via parameter "mass".')
            return None
        binned_mass = kwargs.get("binned_mass")
        if binned_mass is None:
            binned_mass, _, _ = self.value_on_grid(
                pos, mass, statistic="sum", bins=self.n_grid
            )
        binned_pos_list = np.argwhere(binned_mass > 0)
        binned_mass_list = binned_mass[binned_mass > 0]
        return binned_pos_list, binned_mass_list

    def halo_value_on_grid(self, value, **kwargs):
        pos = kwargs.get("pos", self.pos_grid)
        if pos is None:
            print('Provide the halo positions via parameter "pos".')
            return None
        binned_value = kwargs.get("binned_value")
        if binned_value is None:
            binned_value, _, _ = self.value_on_grid(
                pos, value, statistic="sum", bins=self.n_grid
            )
        binned_pos_list = np.argwhere(binned_value > 0)
        binned_value_list = binned_value[binned_value > 0]
        return binned_pos_list, binned_value_list
