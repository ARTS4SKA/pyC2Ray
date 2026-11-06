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
    ExpAccretionParameters,
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
            self.sources_dir / file, self.boxsize
        )

        # source life-time in cgs
        if isinstance(self.parameters.sources.accretion, ExpAccretionParameters):
            # ts = 1. / (self.alph_h * (1+z) * self.cosmology.H(z=z).cgs.value)
            ts = self.fstar_relation.source_lifetime(
                z, self.parameters.sources.accretion.alpha_h
            )
        elif isinstance(self.parameters.sources.accretion, ConstantAccretionParameters):
            assert dt is not None
            ts = dt
        else:
            assert_never()

        # get stellar-to-halo ratio
        if isinstance(self.parameters.sources.fstar, MuvSourceParameters):
            assert isinstance(self.parameters.sources.accretion, ExpAccretionParameters)
            fstar = self.fstar_relation.get(
                Mhalo=srcmass_msun,
                z=z,
                alpha_h=self.parameters.sources.accretion.alpha_h,
                a_s=self.parameters.sources.fstar.a_s,
                b_s=self.parameters.sources.fstar.b_s,
            )
        else:
            fstar = self.fstar_relation.get(Mhalo=srcmass_msun)

        # get escaping fraction
        if isinstance(self.parameters.sources.fesc, ConstantFescParameters):
            fesc = self.parameters.sources.fesc.f0_esc
        elif isinstance(self.parameters.sources.fesc, PowerFescParameters):
            fesc = self.fesc_relation.get(Mhalo=srcmass_msun)
        elif isinstance(self.parameters.sources.fesc, PowerObsFescParameters):
            # here the escaping fraction is fitted to data that uses stellar mass
            fesc = self.fesc_relation.get(Mhalo=fstar * srcmass_msun)
        elif isinstance(self.parameters.sources.fesc, ThesanFescParameters):
            fesc = self.fesc_relation.get(Mhalo=srcmass_msun, z=z)
        else:
            assert_never()

        # get for star formation history
        nr_switchon: int = 0
        if self.acc_model == "exp" and self.bursty_sfr in ("instant", "integrate"):
            burst_mask = self.bursty_relation.get_bursty(mass=srcmass_msun, z=z)

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
            if self.fesc_model == "constant":
                fstar = fstar[burst_mask]
            else:
                assert isinstance(fesc, np.ndarray)
                fstar, fesc = fstar[burst_mask], fesc[burst_mask]
        else:
            # no bursty model
            nr_switchon = srcmass_msun.size
            self.perc_switchon = 100.0

        # if there are sources shitched on then calculate flux
        if nr_switchon > 0:
            # get stellar mass
            mstar_msun = fesc * fstar * srcmass_msun

            # sum together masses into a mesh grid and get a list of the source positon and mass
            srcpos, srcmstar = bin_sources(
                srcpos_mpc=srcpos_mpc,
                mstar_msun=mstar_msun,
                boxsize=self.boxsize / self.cosmology.h,
                meshsize=self.N,
            )

            # normalize flux
            normflux = (
                c.msun2g
                * self.parameters.sources.num_ion
                * srcmstar
                / (c.m_p * ts * S_star_ref)
            )

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
            return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.float64)

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
        file = self.density_dir / fbase
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
            self.density_dir / "redshift_density.txt", usecols=1
        )
        self.zred_sources = np.loadtxt(
            self.sources_dir / "redshift_sources.txt", usecols=1
        )
        if self.resume:
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_dir))
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
        # ext = get_extension_in_folder(path=self.results_dir)
        ext = ".npy"
        if ext == ".dat":
            fname = "%sxfrac_z%.3f.dat" % (self.results_dir, self.zred)
            self.xh = t2c.read_cbin(filename=fname, bits=64, order="F")
            self.phi_ion = t2c.read_cbin(
                filename="%sIonRates_z%.3f.dat" % (self.results_dir, self.zred),
                bits=32,
                order="F",
            )
        elif ext == ".npy":
            fname = self.results_dir / ("xfrac_z%.3f.npy" % self.zred)
            self.xh = np.load(fname)
            self.phi_ion = np.load(
                self.results_dir / ("IonRates_z%.3f.npy" % self.zred)
            )
        else:
            raise FileNotFoundError(
                " Resume file not found: %sxfrac_%.3f.npy"
                % (self.results_dir, self.zred)
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
    def fstar_model(self) -> str:
        return self.parameters.sources.fstar_model

    @property
    def acc_model(self) -> str:
        return self.parameters.sources.accretion_model

    @property
    def bursty_sfr(self) -> str:
        assert isinstance(self.parameters.sources.accretion, ExpAccretionParameters)
        return self.parameters.sources.accretion.bursty_sfr

    @property
    def fesc_model(self) -> str:
        return self.parameters.sources.fesc_model

    def _sources_init(self):
        """Initialize settings to read source files"""
        # --- Stellar-to-Halo Source model ---

        # dictionary with all the f_star parameters
        fstar_pars = self.parameters.sources.fstar

        # print message that inform of the f_star model employed
        if self.fstar_model == "fgamma":
            logger.info(
                f"Using constant stellar-to-halo relation model with f_star = {fstar_pars.f0:.1f}, "
                f"Nion = {self.parameters.sources.num_ion:.1f}"
            )
        elif self.fstar_model in ("dpl", "lognorm"):
            logger.info(
                f"Using {self.fstar_model} to model the stellar-to-halo relation with parameters: {fstar_pars}."
            )
        elif self.fstar_model == "Muv":
            logger.info(
                f"Using {self.fstar_model} to model the stellar-to-halo relation with scatter "
                "and average value with parameters: {fstar_pars}."
            )

        # define the f_star model class (to call self.fstar_model.get_fstar(Mhalo) when reading the sources)
        # same seed on every rank, so that all ranks build the same source field
        rng = np.random.default_rng(self.parameters.sources.seed)

        self.fstar_relation = StellarToHaloRelation(
            fstar_pars, cosmo=self.cosmology, rng=rng
        )

        # --- Halo Accretion Model ---
        # TODO: Create class etc...
        logger.info(f"Using {self.acc_model} accretion to model.")

        # dictionary with all the burstiness parameters
        if self.acc_model == "exp" and self.bursty_sfr in ("instant", "integrate"):
            acc_pars = self.parameters.sources.accretion

            logger.info(
                f"Using {self.bursty_sfr} bustiness to model the star formation history with parameters: {acc_pars}."
            )

            # define the burstiness SF model class
            self.bursty_relation = BurstySFR(
                acc_pars,
                cosmo=self.cosmology,
                rng=rng,
            )
        else:
            logger.info("No bustiness model for the star formation history.")

        # --- Escaping fraction Model ---
        fesc_pars = self.parameters.sources.fesc
        if self.fesc_model == "constant":
            logger.info(
                "Using constant escaping fraction model with f0_esc = %.1f",
                fesc_pars.f0_esc,
            )
        elif self.fesc_model == "power":
            logger.info(
                f"Using mass-dependent power law model for the escaping fraction with parameters: {fesc_pars}"
            )

        self.fesc_relation = EscapeFraction(fesc_pars)
