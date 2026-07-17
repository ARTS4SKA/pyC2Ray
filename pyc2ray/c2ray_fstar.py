import logging
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import tools21cm as t2c

import pyc2ray.constants as c
from pyc2ray.c2ray_base import C2Ray
from pyc2ray.parameters import (
    ConstantAccretionParameters,
    ConstantFescParameters,
    DplSourceParameters,
    ExpAccretionParameters,
    Gelli24FescParameters,
    MuvSourceParameters,
    PowerFescParameters,
    PowerObsFescParameters,
    ThesanFescParameters,
)
from pyc2ray.source_model import BurstySFR, EscapeFraction, StellarToHaloRelation
from pyc2ray.utils import bin_sources
from pyc2ray.utils.other_utils import assert_never, find_bins, get_redshifts_from_output
from pyc2ray.utils.sourceutils import FloatArray, IntArray, PathType

__all__ = ["C2Ray_fstar"]
logger = logging.getLogger(__name__)

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================


class C2Ray_fstar(C2Ray):
    def __init__(self, paramfile: PathType) -> None:
        """Basis class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation
        Nmesh : int
            Mesh size (number of cells in each dimension)
        use_gpu : bool
            Whether to use the GPU-accelerated ASORA library for raytracing

        """
        super().__init__(paramfile)
        logger.info('Running: "C2Ray for %d Mpc/h volume"', self.boxsize)

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def ionizing_flux(
        self,
        file: PathType,
        z: float,
        dt: float | None = None,
        save_Mstar: bool = False,
    ) -> tuple[IntArray, FloatArray]:
        """Read sources from a C2Ray-formatted file
        Parameters
        ----------
        file : Filename to read.
        z : redshift
        dt : time-step in Myrs.
        save_Mstar : whether to save the stellar mass of the sources (not used)


        Returns
        -------
        srcpos : Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : Normalization of the flux of each source (relative to S_star)
        """
        S_star_ref = 1e48

        # read halo list
        srcpos_mpc, srcmass_msun = self.read_haloes(
            self.sources_basename / file, self.boxsize
        )

        # source life-time in cgs
        if isinstance(self.parameters.sources.accretion, ExpAccretionParameters):
            # ts = 1. / (self.alph_h * (1+z) * self.cosmology.H(z=z).cgs.value)
            ts = self.fstar_model.source_lifetime(z=z)
        elif isinstance(self.parameters.sources.accretion, ConstantAccretionParameters):
            assert dt is not None
            ts = dt
        else:
            assert_never()

        # get stellar-to-halo ratio
        if isinstance(self.parameters.sources.fstar, MuvSourceParameters):
            fstar = self.fstar_model.get(
                Mhalo=srcmass_msun,
                z=z,
                a_s=self.parameters.sources.fstar.a_s,
                b_s=self.parameters.sources.fstar.b_s,
            )
        else:
            fstar = self.fstar_model.get(Mhalo=srcmass_msun)

        # get escaping fraction
        if isinstance(self.parameters.sources.fesc, ConstantFescParameters):
            fesc = self.fesc_model.f0_esc
        elif isinstance(self.parameters.sources.fesc, PowerFescParameters):
            fesc = self.fesc_model.get(Mhalo=srcmass_msun)
        elif isinstance(self.parameters.sources.fesc, PowerObsFescParameters):
            # here the escaping fraction is fitted to data that uses stellar mass
            fesc = self.fesc_model.get(Mhalo=fstar * srcmass_msun)
        elif isinstance(self.parameters.sources.fesc, Gelli24FescParameters):
            # mean quantities
            mean_fstar = self.fstar_model.stellar_to_halo_fraction(Mhalo=srcmass_msun)
            mean_Muv = self.fstar_model.UV_magnitude(
                fstar=mean_fstar, mdot=srcmass_msun / ts
            )

            # absolute magnitude with scatter
            Muv = self.fstar_model.UV_magnitude(fstar=fstar, mdot=srcmass_msun / ts)

            # magnitude dependent escaping fraction
            fesc = self.fesc_model.get(delta_Muv=mean_Muv - Muv)
        elif isinstance(self.parameters.sources.fesc, ThesanFescParameters):
            fesc = self.fesc_model.get(Mhalo=srcmass_msun, z=z)
        else:
            assert_never()

        # get for star formation history
        nr_switchon: int = 0
        if self.acc_kind == "exp":
            if self.bursty_sfr == "instant" or self.bursty_sfr == "integrate":
                burst_mask = self.bursty_model.get_bursty(mass=srcmass_msun, z=z)

                nr_switchon = cast(int, np.count_nonzero(burst_mask))
                self.perc_switchon = 100 * nr_switchon / burst_mask.size

                logger.info(
                    " A total of %.2f %% of galaxies (%d out of %d) have bursty star-formation.",
                    self.perc_switchon,
                    nr_switchon,
                    burst_mask.size,
                )

                # mask the sources that are switched off
                srcpos_mpc, srcmass_msun = (
                    srcpos_mpc[burst_mask],
                    srcmass_msun[burst_mask],
                )
                if self.fesc_kind == "constant":
                    fstar = fstar[burst_mask]
                else:
                    fstar, fesc = fstar[burst_mask], fesc[burst_mask]
            else:
                # no bursty model
                nr_switchon = srcmass_msun.size
                self.perc_switchon = 100.0

        assert isinstance(self.parameters.sources.fstar, DplSourceParameters)
        Nion = self.parameters.sources.fstar.Nion
        assert Nion is not None
        # if there are sources shitched on then calculate flux
        if nr_switchon > 0:
            if "spice" in self.fstar_kind:
                # get star formation rate from SPICE tables
                sfr_spice = self.fstar_model.sfr_SPICE(Mhalo=srcmass_msun, z=z)

                # sum together masses into a mesh grid and get a list of the source positon and mass
                srcpos, sfr = bin_sources(
                    srcpos_mpc=srcpos_mpc,
                    mstar_msun=sfr_spice * fesc,
                    boxsize=self.boxsize / self.cosmology.h,
                    meshsize=self.N + 1,
                )

                # normalize flux
                normflux = c.msun2g * Nion * sfr / (c.m_p * S_star_ref)
            else:
                # get stellar mass
                mstar_msun = fesc * fstar * srcmass_msun

                # sum together masses into a mesh grid and get a list of the source positon and mass
                srcpos, srcmstar = bin_sources(
                    srcpos_mpc=srcpos_mpc,
                    mstar_msun=mstar_msun,
                    boxsize=self.boxsize / self.cosmology.h,
                    meshsize=self.N + 1,
                )

                # normalize flux
                normflux = c.msun2g * Nion * srcmstar / (c.m_p * ts * S_star_ref)

            # calculate total number of ionizing photons
            self.tot_phots = np.sum(normflux * dt * S_star_ref)

            logger.info(
                """
---- Reading source file with total of %d ionizing source:
%s
 Total Flux : %e [1/s]
 Total number of ionizing photons : %e
 Source lifetime : %f Myr""",
                normflux.size,
                file,
                np.sum(normflux * S_star_ref),
                self.tot_phots,
                ts / (1e6 * c.year2s),
            )
            if "spice" in self.fstar_kind:
                logger.info(
                    " min, max SFR (grid) : %.3e  %.3e [Msun/yr] and"
                    " min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]",
                    sfr.min() / c.year2s,
                    sfr.max() / c.year2s,
                    normflux.min() * S_star_ref,
                    normflux.mean() * S_star_ref,
                    normflux.max() * S_star_ref,
                )
            else:
                logger.info(
                    " min, max stellar (grid) mass : %.3e  %.3e [Msun] and"
                    " min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]",
                    srcmstar.min(),
                    srcmstar.max(),
                    normflux.min() * S_star_ref,
                    normflux.mean() * S_star_ref,
                    normflux.max() * S_star_ref,
                )

            return srcpos, normflux

        else:
            logger.info(
                """
---- Reading source file with total of %d ionizing source:
%s
 No sources switch on. Skip computing the raytracing.""",
                srcmass_msun.size,
                file,
            )

            self.tot_phots = 0
            return np.array((3, 0), dtype=np.int32), np.array((0,), dtype=np.float64)

    def read_haloes(
        self, halo_file: PathType, box_len: float
    ) -> tuple[IntArray, FloatArray]:
        """Read haloes from a file.

        Parameters
        ----------
        halo_file : Filename to read
        box_len: Length of the box in Mpc/h

        Returns
        -------
        srcpos_mpc : Positions of the haloes in Mpc.
        srcmass_msun : Masses of the haloes in Msun.
        """

        suffix = Path(halo_file).suffix
        if suffix == ".hdf5":
            # Read haloes from a CUBEP3M file format converted in hdf5.
            f = h5py.File(halo_file)
            h = f.attrs["h"]
            srcmass_msun = f["mass"][:] / h  # Msun
            srcpos_mpc = f["pos"][:] / h  # Mpc
            f.close()
        elif suffix == ".dat":
            # Read haloes from a CUBEP3M file format.
            hl = t2c.HaloCubeP3MFull(filename=halo_file, box_len=box_len)
            srcmass_msun = hl.get(var="m") / self.cosmology.h  # Msun
            srcpos_mpc = hl.get(var="pos") / self.cosmology.h  # Mpc
        elif (suffix == ".txt") and ("fof" in str(halo_file)):
            # Read haloes from a PKDGrav converted in txt.
            hl = np.loadtxt(halo_file)
            srcmass_msun = hl[:, 0] / self.cosmology.h  # Msun
            srcpos_mpc = hl[:, 1:] + self.boxsize / 2  # Mpc/h

            # apply periodic boundary condition shift
            srcpos_mpc[srcpos_mpc > self.boxsize] = (
                self.boxsize - srcpos_mpc[srcpos_mpc > self.boxsize]
            )
            srcpos_mpc[srcpos_mpc < 0.0] = self.boxsize + srcpos_mpc[srcpos_mpc < 0.0]
            srcpos_mpc[srcpos_mpc > self.boxsize] = (
                srcpos_mpc[srcpos_mpc > self.boxsize] - self.boxsize
            )

            assert srcpos_mpc.min() >= 0.0
            assert srcpos_mpc.min() <= self.boxsize
            srcpos_mpc /= self.cosmology.h  # Mpc
        elif (suffix == ".txt") and ("halo" in str(halo_file)):
            # Read haloes from a PKDGrav converted in txt (other converntion).
            hl = np.loadtxt(halo_file)
            srcmass_msun = hl[:, 0] / self.cosmology.h  # Msun

            srcpos_mpc = hl[:, 1:]  # Mpc/h

            srcpos_mpc[srcpos_mpc < 0.0] = self.boxsize + srcpos_mpc[srcpos_mpc < 0.0]
            srcpos_mpc[srcpos_mpc > self.boxsize] = (
                srcpos_mpc[srcpos_mpc > self.boxsize] - self.boxsize
            )

            assert srcpos_mpc.min() >= 0.0
            assert srcpos_mpc.min() <= self.boxsize

            srcpos_mpc /= self.cosmology.h  # Mpc
        return srcpos_mpc, srcmass_msun

    def read_density(self, fbase: PathType, z: float) -> None:
        """Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either N-body or hydro-dynamical simulations.
        The field is then smoothed on a coarse mesh grid.

        Parameters
        ----------
        fbase : the file name (without the path) of the file to open
        z : Redshift

        """
        file = self.density_basename / fbase
        if file.suffix == ".npy":
            overd = np.load(file) - 1.0
        else:
            rdr = t2c.Pkdgrav3data(self.boxsize, self.N, Omega_m=self.cosmology.Om0)
            overd = rdr.load_density_field(str(file))

        self.ndens: np.ndarray = (
            self.cosmology.critical_density0.cgs.value
            * self.cosmology.Ob0
            * (1.0 + overd)
            / (self.mean_molecular * c.m_p)
            * (1 + z) ** 3
        )

        # set floor density to avoid cells with value zero
        self.ndens = np.maximum(self.ndens, 5e-6)

        logger.info(
            """
---- Reading density file:
%s
 min, mean and max density : %.3e  %.3e  %.3e [1/cm3]""",
            file,
            self.ndens.min(),
            self.ndens.mean(),
            self.ndens.max(),
        )

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the f_star case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter"""
        # FIXME: HARDCODED FILE NAMES
        self.zred_density = np.loadtxt(
            self.density_basename / "redshift_density.txt", usecols=1
        )
        self.zred_sources = np.loadtxt(
            self.sources_basename / "redshift_sources.txt", usecols=1
        )
        if self.resume:
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename))
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1
            self.zred = self.zred_0

        self.time = self.zred2time(self.zred)

    def _material_init(self):
        """Initialize material properties of the grid"""
        if not self.resume:
            super()._material_init()
            return

        # get extension of the output file
        # ext = get_extension_in_folder(path=self.results_basename)
        ext = ".npy"
        if ext == ".dat":
            fname = "%sxfrac_z%.3f.dat" % (self.results_basename, self.zred)
            self.xh = t2c.read_cbin(filename=fname, bits=64, order="F")
            self.phi_ion = t2c.read_cbin(
                filename="%sIonRates_z%.3f.dat" % (self.results_basename, self.zred),
                bits=32,
                order="F",
            )
        elif ext == ".npy":
            fname = self.results_basename / ("xfrac_z%.3f.npy" % self.zred)
            self.xh = np.load(fname)
            self.phi_ion = np.load(
                self.results_basename / ("IonRates_z%.3f.npy" % self.zred)
            )
        else:
            raise FileNotFoundError(
                " Resume file not found: %sxfrac_%.3f.npy"
                % (self.results_basename, self.zred)
            )

        logger.info(
            """
---- Reading ionized fraction field:
%s
 min, mean and max density : %.5e  %.5e  %.5e""",
            fname,
            self.xh.min(),
            self.xh.mean(),
            self.xh.max(),
        )

        # TODO: implement heating
        self.temp = np.full(self.shape, self.parameters.material.temp0, order="F")

    @property
    def fstar_kind(self) -> str:
        return self.parameters.sources.fstar_kind

    @property
    def acc_kind(self) -> str:
        return self.parameters.sources.accretion_model

    @property
    def bursty_sfr(self) -> str:
        assert isinstance(self.parameters.sources.accretion, ExpAccretionParameters)
        return self.parameters.sources.accretion.bursty_sfr

    @property
    def fesc_kind(self) -> str:
        return self.parameters.sources.fesc_model

    def _sources_init(self):
        """Initialize settings to read source files"""
        # --- Stellar-to-Halo Source model ---

        # dictionary with all the f_star parameters
        fstar = self.parameters.sources.fstar
        fstar_pars = {
            "Nion": fstar.Nion,
            "f0": fstar.f0,
            "Mt": fstar.Mt,
            "Mp": fstar.Mp,
            "g1": fstar.g1,
            "g2": fstar.g2,
            "g3": fstar.g3,
            "g4": fstar.g4,
            "alpha_h": self.parameters.sources.accretion.alpha_h,
        }

        # print message that inform of the f_star model employed
        if self.fstar_kind == "fgamma":
            logger.info(
                f"Using constant stellar-to-halo relation model with f_star = {self.parameters.sources.f0:.1f}, "
                f"Nion = {self.parameters.sources.Nion:.1f}"
            )
        elif self.fstar_kind in ("dpl", "lognorm"):
            logger.info(
                f"Using {self.fstar_kind} to model the stellar-to-halo relation with parameters: {fstar_pars}."
            )
        elif self.fstar_kind == "Muv":
            logger.info(
                f"Using {self.fstar_kind} to model the stellar-to-halo relation with scatter "
                "and average value with parameters: {fstar_pars}."
            )
        elif self.fstar_kind == "spice":
            logger.info(
                f"Using {self.fstar_kind} to model the star formation rate with scatter (Basu+ 2025). "
                "We use a 'dpl' model to define the mean SFR."
            )

        # define the f_star model class (to call self.fstar_model.get_fstar(Mhalo) when reading the sources)
        # FIXME: pass directly the escape fraction parameters
        self.fstar_model = StellarToHaloRelation(
            model=self.fstar_kind, pars=fstar_pars, cosmo=self.cosmology
        )

        # --- Halo Accretion Model ---
        # TODO: Create class etc...
        logger.info(f"Using {self.acc_kind} accretion to model.")

        # dictionary with all the burstiness parameters
        if self.bursty_sfr == "instant" or self.bursty_sfr == "integrate":
            bursty_pars = {
                "beta1": self.parameters.sources.accretion.beta1,
                "beta2": self.parameters.sources.accretion.beta2,
                "tB0": self.parameters.sources.accretion.tB0,
                "tQ_frac": self.parameters.sources.accretion.tQ_frac,
                "z0": self.parameters.sources.accretion.z0,
            }

            logger.info(
                f"Using {self.bursty_sfr} bustiness to model the star formation history with parameters: {bursty_pars}."
            )

            # define the burstiness SF model class
            self.bursty_model = BurstySFR(
                model=self.bursty_sfr,
                pars=bursty_pars,
                alpha_h=self.parameters.sources.alpha_h,
                cosmo=self.cosmology,
            )
        else:
            logger.info("No bustiness model for the star formation history.")

        # --- Escaping fraction Model ---
        fesc_pars = {
            "f0_esc": self.parameters.sources.fesc.f0_esc,
            "Mp_esc": self.parameters.sources.fesc.Mp_esc,
            "al_esc": self.parameters.sources.fesc.al_esc,
        }
        if self.fesc_kind == "constant":
            logger.info(
                "Using constant escaping fraction model with f0_esc = %.1f",
                self.parameters.sources.f0_esc,
            )
        elif self.fesc_kind == "power":
            logger.info(
                f"Using mass-dependent power law model for the escaping fraction with parameters: {fesc_pars}"
            )
        elif self.fesc_kind == "Gelli24":
            logger.info(
                f"Using UV magnitude-dependent power law model for the escaping fraction with parameters: {fesc_pars}"
            )

        # FIXME: pass directly the escape fraction parameters
        self.fesc_model = EscapeFraction(model=self.fesc_kind, pars=fesc_pars)
