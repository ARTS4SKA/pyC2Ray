"""
This file defines the abstract C2Ray object class, which is the basis for a c2ray
simulation. It deals with parameters, I/O, cosmology, and other things such as memory
allocation when using the GPU.  Any concrete simulation uses subclasses of C2Ray, with
methods specific to certain input files (e.g. CubeP3M)

Since all simulation classes inherit from this class, great care should be taken in
editing it!

-- Notes on cosmology: -- * In C2Ray, the scale factor is 1 at z = 0. The box size is
given in comoving units, i.e. it is the proper size at z = 0. At the start (in
cosmo_ini), the cell size & volume are scaled down to the first redshift slice of the
program.

* There are 2 types of redshift evolution: (1) when the program reaches a new "slice"
(where a density file would be read etc) and (2) at each timestep BETWEEN slices.
Basically, at (1), the density is set, and during (2), this density is diluted due to
the expansion.

* During this dilution (at each timestep between slices), C2Ray has the convention that
the redshift is incremented not by the value that corresponds to a full timestep in
cosmic time, but by HALF a timestep.
   ||          |           |           |           |               ||
   ||    z1    |     z2    |     z3    |     z4    |       ...     ||
   ||          |           |           |           |               ||
   t0          t1          t2          t3          t4

  ("||" = slice,    "|" = timestep,   "1,2,3,4,.." indexes the timestep)

In terms of attributes, C2Ray.time always contains the time at the end of the current
timestep, while C2Ray.zred contains the redshift at half the current timestep. This is
relevant to understand the cosmo_evolve routine below (which is based on what is done in
the original C2Ray)

This induces a potential bug: when a slice is reached and the density is set, the
density corresponds to zslice while C2Ray.zred is at the redshift "half a timestep
before".  The best solution I've found here is to just save the comoving cell size dr_c
and always set the current cell size to dr = a(z)*dr_c, rather than "diluting" dr
iteratively like the density.

=======================================================================================

Conversion Factors

When doing direct comparisons with C2Ray, the difference between
astropy.constants and the C2Ray values may be visible, thus we use the same exact value
for the constants.  This can be changed to the astropy values once consistency between
the two codes has been established.
"""

import atexit
import logging
from functools import cached_property
from pathlib import Path
from typing import TypeAlias

import numpy as np
from astropy import constants as cst
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from mpi4py import MPI

import pyc2ray.constants as c
from pyc2ray.asora_core import device_close, device_init, is_device_init, libasora
from pyc2ray.evolve import ChemistryParams, evolve3D
from pyc2ray.parameters import (
    AbundancesParameters,
    BlackBodyParameters,
    CGSParameters,
    CosmologyParameters,
    GridParameters,
    MaterialParameters,
    OutputParameters,
    PhotoParameters,
    RaytracingParameters,
    SinksParameters,
    SourcesParameters,
    YmlParameters,
)
from pyc2ray.radiation import (
    BlackBodyBase,
    BlackBodySource_Multifreq,
    YggdrasilModel,
    make_tau_table,
)
from pyc2ray.sinks_model import SinksPhysics
from pyc2ray.solver.helium import CoolingTables
from pyc2ray.utils.logutils import PathType, configure_logger
from pyc2ray.utils.sourceutils import FloatArray, IntArray

logger = logging.getLogger(__name__)

ParameterClass: TypeAlias = type[YmlParameters]


class C2Ray:
    banner = r"""
                 _________   ____
    ____  __  __/ ____/__ \ / __ \____ ___  __
   / __ \/ / / / /    __/ // /_/ / __ `/ / / /
  / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ /
 / .___/\__, /\____//____/_/ |_|\__,_/\__, /
/_/    /____/                        /____/
"""

    XH_PREFIX = "xfrac"
    PHION_PREFIX = "IonRates"
    PHEAT_PREFIX = "HeatRates"
    TEMP_AV_PREFIX = "Temper"

    def __init__(self, paramfile: PathType) -> None:
        """Basis class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : Name of a YAML file containing parameters for the C2Ray simulation
        """
        # Read YAML parameter file and set main properties
        self._read_paramfile(paramfile)

        # Help type checkers by defining some type annotations
        self.time: float
        self.zred: float
        self.dr: float
        self.xh: tuple[FloatArray, FloatArray, FloatArray]
        self.phion: tuple[FloatArray, FloatArray, FloatArray]
        self.pheat: tuple[FloatArray, FloatArray, FloatArray]
        self.temp_av: FloatArray
        self.clumping_factor: FloatArray
        self.tot_phots: float

        # MPI setup
        if self.mpi:
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.nprocs = MPI.COMM_WORLD.Get_size()
        else:
            self.rank = 0
            self.nprocs = 1

        if self.mpi and MPI.COMM_WORLD.Get_size() <= 1:
            logger.warning(
                "Requested to enable MPI but there is only one process available. "
                "Try to run this application with a higher number of processes. Disabling MPI."
            )
            self.grid_params.mpi = False

        # Set Raytracing mode
        if self.gpu:
            # Initialize the correct GPU device
            shared_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
            local_rank = shared_comm.Get_rank()
            device_init(local_rank)

            # Register deallocation function (automatically calls this on program termination)
            atexit.register(device_close)

        # Initialize output and logger. Waits for all ranks to reach this point.
        self._output_init()

        # Initialize Simulation
        self._grid_init()
        self._cosmology_init()
        self._redshift_init()
        self._material_init()
        self._sources_init()
        self._radiation_init()
        self._sinks_init()

        if self.gpu:
            # Print maximum shell size for info, based on LLS (qmax is s.t. Rmax fits inside of it)
            q_max = np.ceil(np.sqrt(3) * min(self.R_max_LLS, np.sqrt(3) * self.N / 2))
            logger.info(f"Using ASORA Raytracing (q_max = {q_max})")
        else:
            # Print info about subbox algorithm
            logger.info(
                f"Using CPU Raytracing (subboxsize = {self.subboxsize}, max_subbox = {self.max_subbox})"
            )

        # initialize radiation tables
        self._radiation_init()

        if self.mpi:
            MPI.COMM_WORLD.Barrier()
            logger.info(f"Using {self.nprocs} MPI Ranks")
        else:
            logger.info("Running in non-MPI (single-GPU/CPU) mode")

        logger.info("Starting simulation... \n\n")

    # ======================
    # TIME-EVOLUTION METHODS
    # ======================
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

    def cosmo_evolve_to_now(self) -> None:
        """Evolve cosmology over a timestep"""
        # Time step
        t_now = self.time

        # Increment redshift by half a time step
        z_now = self.time2zred(t_now)

        # Scale quantities if cosmological run
        if self.cosmological:
            dilution_factor = (1 + z_now) / (1 + self.zred)

            # Scale density according to expansion
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            self.dr /= dilution_factor

        # Set new time and redshift (after timestep)
        self.zred = z_now

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

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = ((1 + z_half) / (1 + self.zred)) ** 3
            self.ndens *= dilution_factor

            # Set cell size to current proper size
            self.dr = self.dr_c * self.cosmology.scale_factor(z_half)

        # Set new clumping factor if is not redshift constant
        if self.sinks.clumping_model != "constant":
            if self.sinks.clumping_model == "redshift":
                self.clumping_factor = self.sinks.calculate_clumping(z=self.zred)
            else:
                self.clumping_factor = self.sinks.calculate_clumping(
                    z=self.zred, ndens=self.ndens
                )

            logger.info(
                " min, mean and max clumping factor at z = %.3f: %.2f  %.2f  %.2f",
                self.zred,
                self.clumping_factor.min(),
                self.clumping_factor.mean(),
                self.clumping_factor.max(),
            )

        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after

        # Set new mean-free-path if it is redshift dependent
        self.R_max_LLS: float
        if self.sinks.mfp_model == "Worseck2014":
            self.R_max_LLS = self.sinks.mfp_Worseck2014(z=self.zred)  # in cMpc
            self.R_max_LLS *= self.N / self.boxsize  # in number of grids
            logger.info(
                """Mean-free-path for photons at z = %.3f (Worseck+ 2014): %.3e cMpc
This corresponds to %.3f grid cells.""",
                self.zred,
                self.R_max_LLS * self.boxsize / self.N,
                self.R_max_LLS,
            )

    def evolve3D(self, dt: float, src_flux: FloatArray, src_pos: IntArray) -> None:
        """Evolve the grid over one timestep

        Raytrace all sources, compute cumulative photoionization rate of each cell and
        do chemistry. This is done until convergence in the ionized fraction is reached.

        Parameters
        ----------
        dt : Timestep in seconds (typically generated using set_timestep method)
        src_flux : 1D array of shape (numsrc, ) containing the total ionizing flux of each source,
                   normalized by S_star (1e48 by default)
        src_pos : 2D array of shape (3, numsrc) containing the 3D grid position of each source,
                  in Fortran indexing (from 1)
        """
        if src_pos.shape[0] != 3:
            src_pos = src_pos.T

        NumSrc = len(src_flux)
        if len(src_flux) != src_pos.shape[1]:
            raise ValueError(
                "ASORA requires the shape of src_pos to be (3, num_src) and the shape of src_num to be (num_src, )."
            )

        # If the number of sources exceed the number of MPI processors
        # then call the evolve designed for the MPI source splitting.
        # Otherwise all ranks are calling (independently) the evolve
        # with no source splitting until the condition above is meet.
        use_mpi = NumSrc >= self.nprocs and self.mpi

        self.xh, self.phion, self.pheat, self.temp_av = evolve3D(
            Hz=self.cosmology.H(self.zred).cgs.value,
            dt=dt,
            dr=self.dr,
            R_max=self.R_max_LLS,
            src_flux=src_flux,
            src_pos=src_pos,
            src_batch_size=self.raytracing_params.source_batch_size,
            temp=self.temp_av,
            ndens=self.ndens,
            clump=self.clumping_factor,
            xh=self.xh,
            chems=self.chem_parms,
            logtau=(self.minlogtau, self.dlogtau, len(self.photo_thin_table)),
            logtemp=self.cool_tables.logtemp,
            convergence_fraction=self.convergence_fraction,
            use_gpu=self.gpu,
            use_mpi=use_mpi,
            rank=self.rank if use_mpi else 0,
            nprocs=self.nprocs if use_mpi else 1,
            # c2ray only parameters
            photo_thin_table=self.photo_thin_table,
            photo_thick_table=self.photo_thick_table,
            sigma=self.sigma,
            max_subbox=self.max_subbox,
            subboxsize=self.subboxsize,
            loss_fraction=self.loss_fraction,
        )

    def write_output(
        self,
        z: float,
        log_history: bool = True,
        write_summary: bool = True,
    ) -> None:
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z :
            Redshift (used to name the file)
        log_history:
            Log the min, mean, and max of the ionization fractions, rates, and temperature.
        write_summary :
            Write a summary of the simulation to a text file.
        """
        if self.rank != 0:
            return

        self._write_grids(z)
        if log_history:
            self._log_history()
        if write_summary:
            self._write_summary(z)

    def _write_grids(self, z: float) -> None:
        def save_npz(
            prefix: str, data: tuple[FloatArray, FloatArray, FloatArray]
        ) -> None:
            nonlocal z
            filename = self.results_basename / f"{prefix}_z{z:.3f}.npz"
            np.savez(filename, HII=data[0], HeII=data[1], HeIII=data[2])

        save_npz(C2Ray.XH_PREFIX, self.xh)
        save_npz(C2Ray.PHION_PREFIX, self.phion)
        save_npz(C2Ray.PHEAT_PREFIX, self.pheat)
        np.savez(
            self.results_basename / f"{C2Ray.TEMP_AV_PREFIX}_z{z:.3f}.npz",
            temp_av=self.temp_av,
        )

    def _log_history(self) -> None:
        # Prevent expensive computation of stats if logger is not enabled.
        if not logger.isEnabledFor(logging.INFO):
            return

        def min_mean_max(data: FloatArray, label: str, units: str = "") -> str:
            return f"min, mean, max {label} : {data.min():.5e}, {data.mean():.5e}, {data.max():.5e} [{units}]\n"

        logger.info(
            "\n--- Reionization History ----\n"
            + min_mean_max(self.xh[0], "xHII")
            + min_mean_max(self.xh[1], "xHeII")
            + min_mean_max(self.xh[2], "xHeIII")
            + min_mean_max(self.phion[0], "Irate (HI)", "1/s")
            + min_mean_max(self.phion[1], "Irate (HeI)", "1/s")
            + min_mean_max(self.phion[2], "Irate (HeII)", "1/s")
            + min_mean_max(self.pheat[0], "Hrate (HI)", "1/s")
            + min_mean_max(self.pheat[1], "Hrate (HeI)", "1/s")
            + min_mean_max(self.pheat[2], "Hrate (HeII)", "1/s")
            + min_mean_max(self.temp_av, "Temperature", "K")
            + min_mean_max(self.ndens, "Density", "1/cm3")
        )

    def _write_summary(self, z: float) -> None:
        with open(self.results_basename / "PhotonCounts2.txt", "a") as f:
            # File is empty, write header
            if f.tell() == 0:
                header = (
                    "# z\t"
                    "tot HI atoms\t"
                    "tot HeI atoms\t"
                    "tot HeII atoms\t"
                    "tot phots\t"
                    "mean ndens [1/cm3]\t"
                    "mean Irate HI [1/s]\t"
                    "mean Irate HeI [1/s]\t"
                    "mean Irate HeII [1/s]\t"
                    "mean Hrate HI [1/s]\t"
                    "mean Hrate HeI [1/s]\t"
                    "mean Hrate HeII [1/s]\t"
                    "mean temperature [K]\t"
                    "R_mfp [cMpc]\t"
                    "mean HII by volume and mass\n"
                )
                f.write(header)

            tot_n = self.ndens * self.dr**3
            tot_nHI = tot_n * (1 - self.xh[0]).sum()
            tot_nHeI = tot_n * (1 - self.xh[1] - self.xh[2]).sum()
            tot_nHeII = tot_n * self.xh[1].sum()

            R_mfp = self.R_max_LLS / self.N * self.boxsize
            massavg_ion_frac = (self.xh[0] * self.ndens).sum() / self.ndens.sum()

            ion_HI, ion_HeI, ion_HeII = self.phion
            heat_HI, heat_HeI, heat_HeII = self.pheat

            text = (
                f"{z:.3f}\t"
                f"{tot_nHI:.3e}\t"
                f"{tot_nHeI:.3e}\t"
                f"{tot_nHeII:.3e}\t"
                f"{self.tot_phots:.3e}\t"
                f"{self.ndens.mean():.3e}\t"
                f"{ion_HI.mean():.3e}\t"
                f"{ion_HeI.mean():.3e}\t"
                f"{ion_HeII.mean():.3e}\t"
                f"{heat_HI.mean():.3e}\t"
                f"{heat_HeI.mean():.3e}\t"
                f"{heat_HeII.mean():.3e}\t"
                f"{self.temp_av.mean():.3e}\t"
                f"{R_mfp:.3e}\t"
                f"{massavg_ion_frac:.3e}\n"
            )
            f.write(text)

    # ===============
    # UTILITY METHODS
    # ===============

    def time2zred(self, t: float) -> float:
        """Calculate the redshift corresponding to an age t in seconds"""
        return z_at_value(self.cosmology.age, t * u.s).value

    def zred2time(self, z: float, unit: str = "s") -> float:
        """Calculate the age corresponding to a redshift z

        Parameters
        ----------
        z : Redshift at which to get age
        unit : Unit to get age in astropy naming. Default: seconds
        """
        return self.cosmology.age(z).to(unit).value

    # TODO: figure out if all these property methods are necessary
    @property
    def N(self) -> int:
        return self.grid_params.meshsize

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.N, self.N, self.N

    @property
    def boxsize(self) -> float:
        return self.grid_params.boxsize

    @property
    def gpu(self) -> bool:
        return bool(self.grid_params.gpu)

    @property
    def mpi(self) -> bool:
        return bool(self.grid_params.mpi)

    @property
    def resume(self) -> bool:
        return bool(self.grid_params.resume)

    @property
    def eth0(self) -> float:
        return self.cgs_params.eth0

    @property
    def ethe0(self) -> float:
        return self.cgs_params.ethe0

    @property
    def ethe1(self) -> float:
        return self.cgs_params.ethe1

    @property
    def fh0(self) -> float:
        return self.cgs_params.fh0

    @property
    def xih0(self) -> float:
        return self.cgs_params.xih0

    @property
    def abu_h(self) -> float:
        return self.abundance_params.abu_h

    @property
    def abu_he(self) -> float:
        return self.abundance_params.abu_he

    @property
    def mean_molecular(self) -> float:
        return self.abundance_params.mean_molecular

    @property
    def sigma(self) -> float:
        return self.photo_params.sigma_HI_at_ion_freq

    @property
    def loss_fraction(self) -> float:
        return self.raytracing_params.loss_fraction

    @property
    def convergence_fraction(self) -> float:
        return self.raytracing_params.convergence_fraction

    @property
    def max_subbox(self) -> int:
        return self.raytracing_params.max_subbox

    @property
    def subboxsize(self) -> int:
        return self.raytracing_params.subboxsize

    @property
    def cosmological(self) -> bool:
        return self.cosmology_params.cosmological

    @cached_property
    def chem_parms(self) -> ChemistryParams:
        return ChemistryParams(
            self.cgs_params.bh00,
            self.cgs_params.albpow,
            self.cgs_params.colh0,
            self.cgs_params.temph0,
            self.abundance_params.abu_c,
        )

    @property
    def minlogtau(self) -> float:
        return self.photo_params.minlogtau

    @property
    def maxlogtau(self) -> float:
        return self.photo_params.maxlogtau

    @property
    def NumTau(self) -> int:
        return self.photo_params.NumTau

    @property
    def SourceType(self) -> str:
        return self.photo_params.SourceType

    @property
    def grey(self) -> bool:
        return self.photo_params.grey

    @property
    def compute_heating_rates(self) -> bool:
        return self.photo_params.compute_heating_rates

    @property
    def cs_pl_idx_h(self) -> float:
        return self.blackbody_params.cross_section_pl_index

    @property
    def bb_Teff(self) -> float:
        return self.blackbody_params.Teff

    @property
    def results_basename(self) -> Path:
        return Path(self.output_params.results_basename)

    @property
    def inputs_basename(self) -> Path:
        assert self.output_params.inputs_basename is not None
        return Path(self.output_params.inputs_basename)

    @property
    def sources_basename(self) -> Path:
        assert self.output_params.sources_basename is not None
        return Path(self.output_params.sources_basename)

    @property
    def density_basename(self) -> Path:
        assert self.output_params.density_basename is not None
        return Path(self.output_params.density_basename)

    @property
    def logfile(self) -> Path:
        return self.results_basename / self.output_params.logfile

    # ======================
    # INITIALIZATION METHODS
    # ======================

    def _cosmology_init(self) -> None:
        """Set up cosmology from parameters (H0, Omega,..)"""
        h = self.cosmology_params.h
        Om0 = self.cosmology_params.Omega0
        Ob0 = self.cosmology_params.Omega_B
        Tcmb0 = self.cosmology_params.cmbtemp
        H0 = 100 * h
        self.cosmology = FlatLambdaCDM(H0, Om0, Tcmb0, Ob0=Ob0)
        self.zred_0 = self.cosmology_params.zred_0

        self.age_0 = self.zred2time(self.zred_0)

        # Scale quantities to the initial redshift
        if self.cosmological:
            logger.info(f"""Cosmology is on, scaling comoving quantities to the initial redshift, which is z0 = {self.zred_0:.3f}...
Cosmological parameters used:
h   = {h:.4f}, Tcmb0 = {Tcmb0:.3e}
Om0 = {Om0:.4f}, Ob0   = {Ob0:.4f}""")
            self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
        else:
            logger.info("Cosmology is off.")

    def _radiation_init(self) -> None:
        """Set up radiation tables for ionization/heating rates"""
        # Create optical depth table (log-spaced)

        if self.grey:
            logger.info("Warning: Using grey opacity")
        else:
            logger.info(
                f"Using power-law opacity with {self.NumTau:n} table points between tau=10^({self.minlogtau:n}) "
                f"and tau=10^({self.maxlogtau:n})"
            )

        # The actual table has NumTau + 1 points: the 0-th position is tau=0 and
        # the remaining NumTau points are log-spaced from minlogtau to maxlogtau (same as in C2Ray)
        self.tau, self.dlogtau = make_tau_table(
            self.minlogtau, self.maxlogtau, self.NumTau
        )

        ion_freq_HI = c.ev2fr * self.eth0
        ion_freq_HeII = c.ev2fr * self.ethe1

        radsource: BlackBodyBase
        # Black-Body source type
        if self.SourceType == "blackbody":
            freq_min = ion_freq_HI
            freq_max = 10 * ion_freq_HeII

            # Initialize spectrum parameters
            radsource = BlackBodySource_Multifreq(self.bb_Teff, self.grey)

            logger.info(f"""Using Black-Body sources with effective temperature T = {radsource.temp:.1e} K and Radius {(radsource.R_star / cst.R_sun.to("cm")).value: .3e} rsun
Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz
This is Energy:           {freq_min / c.ev2fr:.3e} to {freq_max / c.ev2fr:.3e} eV""")
        elif self.SourceType == "powerlaw":
            # TODO: power law spectra is already implemented in radiation folder
            pass
        elif self.SourceType == "Zackrisson2011":
            freq_min = ion_freq_HI
            freq_max = 10 * ion_freq_HI  # maximum frequency in Zackrisson tables

            fname = self.photo_params.sed_table
            radsource = YggdrasilModel(
                tabname=fname,
                grey=self.grey,
                freq0=ion_freq_HI,
                pl_index=self.cs_pl_idx_h,
                S_star_ref=1e48,
            )

            logger.info(
                """Using Yggdrasil Models for SED, Zackrisson et al (2011), for PopIII or PopII sources
Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz
This is Energy:           {freq_min / c.ev2fr:.3e} to {freq_max / c.ev2fr:.3e} eV"""
            )
        else:
            raise NameError("Unknown source type : ", self.SourceType)

        # Integrate table
        logger.info("Integrating photoionization rate tables...")
        self.photo_thin_table, self.photo_thick_table = radsource.make_photo_table(
            self.tau, freq_min, freq_max, 1e48
        )

        if self.compute_heating_rates:
            logger.info("Integrating photoheating rate tables...")
            self.heat_thin_table, self.heat_thick_table = radsource.make_heat_table(
                self.tau, freq_min, freq_max, 1e48
            )  # nb integration bounds are given in log10(freq/freq_HI)
        else:
            logger.warning("No heating rates")
            self.heat_thin_table = np.zeros_like(self.photo_thin_table)
            self.heat_thick_table = np.zeros_like(self.photo_thick_table)

        self.cool_tables = CoolingTables.from_dir()

        # Copy radiation table to GPU
        if self.gpu:
            assert is_device_init()
            assert libasora is not None

            libasora.photo_tables_to_device(
                self.photo_thin_table,
                self.photo_thick_table,
                self.heat_thin_table,
                self.heat_thick_table,
            )
            logger.info("Successfully copied radiation tables to GPU memory.")

            libasora.cooling_tables_to_device(*self.cool_tables.astuple())
            logger.info("Successfully copied cooling tables to GPU memory.")

    def _grid_init(self) -> None:
        """Set up grid properties"""
        # Comoving quantities
        self.boxsize_c = self.boxsize * c.Mpc
        self.dr_c = self.boxsize_c / self.N

        logger.info(
            f"""Welcome! Mesh size is N = {self.N:n}.
Simulation Box size (comoving Mpc): {self.boxsize:.3e}"""
        )

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # TODO: need to give the index of start for the redshift loop in the main

    def _output_init(self) -> None:
        """Set up output & log file"""
        # Create result folder
        if self.rank == 0:
            self.results_basename.mkdir(parents=True, exist_ok=True)
            # If it's a new job, delete the old logfile
            if not self.resume:
                self.logfile.unlink(missing_ok=True)

        # Wait here for result directory to be created
        if self.mpi:
            MPI.COMM_WORLD.Barrier()

        configure_logger(self.logfile)

        if self.resume:
            title = f"\n\nResuming{C2Ray.banner[8:]}\n\n"
        else:
            title = f"{C2Ray.banner}\nLog file for pyC2Ray.\n\n"

        logger.info(title)

    def _sinks_init(self) -> None:
        """Initialize sinks physics class for the mean-free path and clumping factor"""

        # init sink physics class for MFP and clumping
        self.sinks = SinksPhysics(self.sinks_params, self.N, self.boxsize)

        # for clumping factor
        if self.sinks.clumping_model == "constant":
            self.clumping_factor = self.sinks.clumping_factor
        elif self.sinks.clumping_model == "redshift":
            self.clumping_factor = self.sinks.calculate_clumping(z=self.zred_0)
        else:
            self.clumping_factor = self.sinks.calculate_clumping(
                z=self.zred_0, ndens=self.ndens
            )

        logger.info(
            """
---- Calculated Clumping Factor (%s model):
 min, mean and max clumping : %.3e  %.3e  %.3e""",
            self.sinks.clumping_model,
            self.clumping_factor.min(),
            self.clumping_factor.mean(),
            self.clumping_factor.max(),
        )
        # for mean-free-path
        if self.sinks.mfp_model == "constant":
            # Set R_max (LLS 3) in cell units
            self.R_max_LLS = self.sinks.R_mfp_cell_unit
            logger.info(
                """
---- Calculated Mean-Free Path (%s model):
Maximum comoving distance for photons from source mfp = %.2f cMpc (%s model).
This corresponds to %.3f grid cells.
""",
                self.sinks.mfp_model,
                self.R_max_LLS * self.boxsize / self.N,
                self.sinks.mfp_model,
                self.R_max_LLS,
            )
        elif self.sinks.mfp_model == "Worseck2014":
            # set mean-free-path to the initial redshift
            self.R_max_LLS = self.sinks.mfp_Worseck2014(z=self.zred_0)  # in cMpc
            self.R_max_LLS *= self.N / self.boxsize
            logger.info(
                """
---- Calculated Mean-Free Path (%s model):
Maximum comoving distance for photons from source mfp = %.2f cMpc (%s model) : A = %.2f Mpc, eta = %.2f.
This corresponds to %.3f grid cells.
""",
                self.sinks.mfp_model,
                self.R_max_LLS * self.boxsize / self.N,
                self.sinks.mfp_model,
                self.sinks.A_mfp,
                self.sinks.etha_mfp,
                self.R_max_LLS,
            )

    # The following initialization methods are simulation kind-dependent and need to be overridden in the subclasses
    def _redshift_init(self) -> None:
        """Initialize time and redshift counter"""
        self.zred = self.zred_0
        self.time = self.zred2time(self.zred)

    def _material_init(self) -> None:
        """Initialize material properties of the grid"""
        self.ndens = np.empty(self.shape, dtype=np.float64, order="F")
        self.xh = (
            np.full_like(self.ndens, self.material_params.xHII),
            np.full_like(self.ndens, self.material_params.xHeII),
            np.full_like(self.ndens, self.material_params.xHeIII),
        )
        self.phion = (
            np.zeros_like(self.ndens),
            np.zeros_like(self.ndens),
            np.zeros_like(self.ndens),
        )
        self.pheat = (
            np.zeros_like(self.ndens),
            np.zeros_like(self.ndens),
            np.zeros_like(self.ndens),
        )
        self.temp_av = np.full_like(self.ndens, self.material_params.temp0)

    def _sources_init(self) -> None:
        """Initialize settings to read source files"""
        pass

    def _read_paramfile(self, paramfile: PathType) -> None:
        """Read in YAML parameter file"""
        ld = YmlParameters.load_yaml(paramfile)

        self.output_params = OutputParameters.from_yml(ld)
        self.grid_params = GridParameters.from_yml(ld)
        self.raytracing_params = RaytracingParameters.from_yml(ld)
        self.material_params = MaterialParameters.from_yml(ld)
        self.cgs_params = CGSParameters.from_yml(ld)
        self.cosmology_params = CosmologyParameters.from_yml(ld)
        self.abundance_params = AbundancesParameters.from_yml(ld)
        self.photo_params = PhotoParameters.from_yml(ld)
        self.sinks_params = SinksParameters.from_yml(ld)
        self.blackbody_params = BlackBodyParameters.from_yml(ld)
        self.sources_params = SourcesParameters.from_yml(ld)
