import logging
from pathlib import Path

import h5py
import numpy as np
import tools21cm as t2c

import pyc2ray.constants as c

from .c2ray_base import C2Ray
from .utils import get_source_redshifts
from .utils.other_utils import find_bins, get_redshifts_from_output
from .utils.sourceutils import FloatArray, IntArray, PathType

__all__ = ["C2Ray_244Test"]

logger = logging.getLogger(__name__)

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================


class C2Ray_244Test(C2Ray):
    def __init__(
        self, paramfile: PathType, Nmesh: int, use_gpu: bool, use_mpi: bool
    ) -> None:
        """Basis class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : Name of a YAML file containing parameters for the C2Ray simulation
        Nmesh : Mesh size (number of cells in each dimension)
        use_gpu : Whether to use the GPU-accelerated ASORA library for raytracing

        """
        super().__init__(paramfile)
        logger.info('Running: "C2Ray for 244 Mpc/h test"')

        self.prev_zdens: float
        self.prev_zsourc: float

    # =====================================================================================================
    # TIME-EVOLUTION METHODS
    # =====================================================================================================
    def set_timestep(self, z1: float, z2: float, num_timesteps: int) -> float:
        """Compute timestep to use between redshift slices

        Parameters
        ----------
        z1 : Initial redshift
        z2 : Next redshift
        num_timesteps : Number of timesteps between the two slices

        Returns
        -------
        dt : Timestep to use in seconds
        """
        t2 = self.zred2time(z2)
        t1 = self.zred2time(z1)
        dt = (t2 - t1) / num_timesteps
        return dt

    def cosmo_evolve(self, dt: float) -> None:
        """Evolve cosmology over a timestep

        Note that if cosmological is set to false in the parameter file, this
        method does nothing!

        Following the C2Ray convention, we set the redshift according to the
        half point of the timestep.
        """
        # Time step
        t_now = self.time
        t_half = t_now + 0.5 * dt
        t_after = t_now + dt
        logger.info(f" This is time : {t_now / c.year2s}\t{t_after / c.year2s}")

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = (1 + z_half) / (1 + self.zred)
            # dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            # self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr /= dilution_factor
            logger.info(f"zfactor = {1.0 / dilution_factor: .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after

    # TODO: factorize with above method
    def cosmo_evolve_to_now(self) -> None:
        """Evolve cosmology over a timestep"""
        # Time step
        t_now = self.time

        # Increment redshift by half a time step
        z_now = self.time2zred(t_now)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = (1 + z_now) / (1 + self.zred)
            # dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            # self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr /= dilution_factor
            logger.info(f"zfactor = {1.0 / dilution_factor: .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_now

    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self, t: float) -> float:
        """Calculate the redshift corresponding to an age t in seconds"""
        # TODO: it should be then z_at_value(self.cosmology.age, t*u.s).value
        # in C2Ray is defined: time2zred = -1+(1.+zred_t0)*(t0/(t0+time))**(2./3.)
        # return -1+(1.+self.zred_0)*(self.age_0/(self.age_0+t))**(2./3.)
        return -1 + (1.0 + self.zred_0) * (self.age_0 / (t)) ** (2.0 / 3.0)

    def zred2time(self, z: float, unit: str = "s") -> float:
        """Calculate the age corresponding to a redshift z

        Parameters
        ----------
        z : Redshift at which to get age
        unit : Unit to get age in astropy naming. Default: seconds
        """
        # TODO : it should be then self.cosmology.age(z).to(unit).value
        # In C2Ray is defined: zred2time = t0*( ((1.0+zred_t0)/(1.0+zred1))**1.5 - 1.0 )
        # return self.age_0*(((1.0+self.zred_0)/(1.0+z))**1.5 - 1.0)
        # C2Ray version, time is 0 at sim begin
        return self.age_0 * (
            ((1.0 + self.zred_0) / (1.0 + z)) ** 1.5
        )  # <- Here, we want time to be actual age (age0 + t)

    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

    def _cosmology_init(self) -> None:
        """Set up cosmology from parameters (H0, Omega,..)"""
        super()._cosmology_init()

        # TODO: it should be:
        # self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
        self.dr = self.dr_c / (1 + self.zred_0)

        H0 = 100 * self.cosmology_params.h
        Om0 = self.cosmology_params.Omega0
        # H0 *= 1e5/c.Mpc

        # self.age_0 = 2.*(1.+self.zred_0)**(-1.5)/(3.*H0*np.sqrt(Om0))
        # self.age_0 = self.zred2time(self.zred_0)
        self.age_0 = (
            2.0
            * (1.0 + self.zred_0) ** (-1.5)
            / (3.0 * H0 * 1e5 / c.Mpc * np.sqrt(Om0))
        )

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def read_sources(
        self, file: PathType, mass: float, ts: float
    ) -> tuple[IntArray, FloatArray]:
        """Read sources from a C2Ray-formatted file

        Parameters
        ----------
        file : Filename to read
        mass: Mass of the sources in Msun (used to compute the flux normalization)
        ts: Lifetime of the sources in Myr (used to compute the flux normalization)

        Returns
        -------
        srcpos : Grid positions of the sources formatted in a suitable way for the chosen
            raytracing algorithm
        normflux : Normalization of the flux of each source (relative to S_star)
        """
        S_star_ref = 1e48

        # TODO: automatic selection of low mass or high mass.
        # For the moment only high mass:
        # mass2phot = (
        #     c.msun2g
        #     * self.fgamma_hm
        #     * self.cosmology.Ob0
        #     / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)
        # )
        # TODO: for some reason the difference with the orginal Fortran run is of
        # the molecular weight: logger.info(str(self.mean_molecular))
        mass2phot = (
            c.msun2g
            * self.fgamma_hm
            * self.cosmology.Ob0
            / (c.m_p * ts * self.cosmology.Om0)
        )

        if Path(file).suffix == ".hdf5":
            f = h5py.File(file, "r")
            srcpos = f["sources_positions"][:].T
            assert srcpos.shape[0] == 3
            normflux = f["sources_mass"][:] * mass2phot / S_star_ref
            f.close()
        else:
            # use original C2Ray source file
            src = t2c.SourceFile(filename=file, mass=mass)
            srcpos = src.sources_list[:, :3].T
            normflux = src.sources_list[:, -1] * mass2phot / S_star_ref

        logger.info(
            """
---- Reading source file with total of %d ionizing source:
%s
 Total Flux : %e
 Source lifetime : %f Myr
 min, max source mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]""",
            normflux.size,
            file,
            np.sum(normflux * S_star_ref),
            ts / (1e6 * c.year2s),
            normflux.min() / mass2phot * S_star_ref,
            normflux.max() / mass2phot * S_star_ref,
            normflux.min() * S_star_ref,
            normflux.mean() * S_star_ref,
            normflux.max() * S_star_ref,
        )
        return srcpos, normflux

    def read_density(self, z: float) -> None:
        """Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either
        N-body or hydro-dynamical simulations. The field is then smoothed
        on a coarse mesh grid.

        Parameters
        ----------
        n : int
            Number of sources to read from the file

        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for
            the chosen raytracing algorithm
        normflux : array
            density mesh-grid in csg units
        """
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0

        # redshift bin for the current redshift based on the density redshift
        # low_z, high_z = find_bins(redshift, self.zred_density)
        high_z = self.zred_density[
            np.argmin(
                np.abs(self.zred_density[self.zred_density >= redshift] - redshift)
            )
        ]

        if high_z != self.prev_zdens:
            file = f"{self.inputs_basename}coarser_densities/{high_z:.3f}n_all.dat"
            self.ndens = (
                t2c.DensityFile(filename=file).cgs_density
                / (self.mean_molecular * c.m_p)
                * (1 + redshift) ** 3
            )
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
            self.prev_zdens = high_z
        else:
            # no need to re-read the same file again
            # TODO: in the future use this values for a 3D interpolation
            # for the density (can be extended to sources too)
            pass

    def write_output(self, z: float, ext: str = ".dat") -> None:
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z : Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.dat"
        t2c.save_cbin(
            filename=self.results_basename / f"xfrac{suffix}",
            data=self.xh,
            bits=64,
            order="F",
        )
        t2c.save_cbin(
            filename=self.results_basename / f"IonRates{suffix}",
            data=self.phi_ion,
            bits=32,
            order="F",
        )

        logger.info(
            """
--- Reionization History ----
 min, mean, max xHII : %.5e  %.5e  %.5e
 min, mean, max Irate : %.5e  %.5e  %.5e [1/s]
 min, mean, max density : %.5e  %.5e  %.5e [1/cm3]""",
            self.xh.min(),
            self.xh.mean(),
            self.xh.max(),
            self.phi_ion.min(),
            self.phi_ion.mean(),
            self.phi_ion.max(),
            self.ndens.min(),
            self.ndens.mean(),
            self.ndens.max(),
        )

    # =============================================================================
    # Below are the overridden initialization routines specific to the CubeP3M case
    # =============================================================================

    def _redshift_init(self) -> None:
        """Initialize time and redshift counter"""
        self.zred_density = t2c.get_dens_redshifts(
            self.inputs_basename + "coarser_densities/"
        )[::-1]
        # self.zred_sources = get_source_redshifts(self.inputs_basename + "sources/")[
        #     ::-1
        # ]
        # TODO: waiting for next tools21cm release
        self.zred_sources = get_source_redshifts(self.inputs_basename + "sources/")[
            ::-1
        ]
        if self.resume:
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename))
            # self.age_0 = self.zred2time(self.zred_0)
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1
            self.zred = self.zred_0

        self.time = self.zred2time(self.zred)
        # self.time = self.age_0

    def _material_init(self) -> None:
        """Initialize material properties of the grid"""
        if self.resume:
            # get fields at the resuming redshift
            self.ndens = (
                t2c.DensityFile(
                    filename="{self.inputs_basename}coarser_densities/{self.prev_zdens:.3f}n_all.dat"
                ).cgs_density
                / (self.mean_molecular * c.m_p)
                * (1 + self.zred) ** 3
            )
            # self.ndens = self.read_density(z=self.zred)
            self.xh = t2c.read_cbin(
                filename="%sxfrac_%.3f.dat" % (self.results_basename, self.zred),
                bits=64,
                order="F",
            )
            # TODO: implement heating
            self.temp = np.full(self.shape, self.material_params.temp0, order="F")
            self.phi_ion = t2c.read_cbin(
                filename="%sIonRates_%.3f.dat" % (self.results_basename, self.zred),
                bits=32,
                order="F",
            )
        else:
            super()._material_init()

    @property
    def fgamma_hm(self) -> float:
        assert self.sources_params.fgamma_hm is not None
        return self.sources_params.fgamma_hm

    @property
    def fgamma_lm(self) -> float:
        assert self.sources_params.fgamma_lm is not None
        return self.sources_params.fgamma_lm

    @property
    def ts(self) -> float:
        assert self.sources_params.ts is not None
        return self.sources_params.ts * c.year2s * 1e6

    def _sources_init(self) -> None:
        """Initialize settings to read source files"""
        logger.info(
            f"Using UV model with fgamma_lm = {self.fgamma_lm:.1f} "
            f"and fgamma_hm = {self.fgamma_hm:.1f}"
        )

    def _grid_init(self) -> None:
        """Set up grid properties"""
        # Comoving quantities
        self.boxsize_c = self.boxsize * c.Mpc / self.cosmology_params.h
        self.dr_c = self.boxsize_c / self.N

        logger.info(f"Welcome! Mesh size is N = {self.N:n}.")
        logger.info(f"Simulation Box size (comoving Mpc): {self.boxsize_c / c.Mpc:.3e}")

        # Initialize cell size to comoving size (if cosmological run,
        # it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # Set R_max (LLS 3) in cell units
        assert self.sinks_params.R_max_cMpc is not None
        self.R_max_LLS = (
            self.sinks_params.R_max_cMpc
            * self.N
            * self.cosmology_params.h
            / self.grid_params.boxsize
        )
        logger.info(f"""Maximum comoving distance for photons from source (type 3 LLS): 
{self.sinks_params.R_max_cMpc: .3e} comoving Mpc
This corresponds to {self.R_max_LLS: .3f} grid cells.""")
