import logging
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import tools21cm as t2c
from astropy import constants as cst

import pyc2ray.constants as c
from pyc2ray.radiation.zbinned_tables import (
    BBFittedPhotoTableSet,
    BPASSPhotoTableSet,
    BPASSQionGrid,
)

from .c2ray_base import C2Ray
from .source_model import BurstySFR, EscapeFraction, StellarToHaloRelation
from .parameters import BPASSParameters, MetallicityEvolutionParameters
from .utils import bin_sources
from .utils.other_utils import (
    find_bins,
    get_extension_in_folder,
    get_redshifts_from_output,
)
from .utils.sourceutils import FloatArray, IntArray, PathType

from pyc2ray.asora_core import photo_table_to_device
from pyc2ray.radiation import BlackBodySource, BPASSSource

# Additional imports for metallicity evolution
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from numpy.lib import recfunctions as rfn
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import yaml
from tqdm import tqdm
import astropy.units as u
import astropy.cosmology.units as cu
from typing import Optional

__all__ = ["C2Ray_Metals"]
logger = logging.getLogger(__name__)


class C2Ray_Metals(C2Ray):
    def __init__(self, paramfile: PathType) -> None:
        """Class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation
        """
        super().__init__(paramfile)
        logger.info('Running: "C2Ray_Metals for %d Mpc/h volume"', self.boxsize)

        # Gridded stellar mass of the current slice's sources (set by
        # ionizing_flux; consumed by the q_ion normalisation scenarios).
        self.src_mstar: np.ndarray | None = None



    # =====================================================================================================
    # USER DEFINED METHODS
    # ====================================================================================================

    def ionizing_flux(
        self,
        file: PathType,
        z: float,
        dt: float | None = None,
        save_Mstar: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            f"{self.sources_basename}{file}", self.boxsize
        )

        # source life-time in cgs
        if self.acc_kind == "EXP":
            # ts = 1. / (self.alph_h * (1+z) * self.cosmology.H(z=z).cgs.value)
            ts = self.fstar_model.source_lifetime(z=z)
        elif self.acc_kind == "constant":
            assert dt is not None
            ts = dt

        # get stellar-to-halo ratio
        if self.fstar_kind == "Muv":
            fstar = self.fstar_model.get(
                Mhalo=srcmass_msun,
                z=z,
                a_s=self.sources_params.a_s,
                b_s=self.sources_params.b_s,
            )
        else:
            fstar = self.fstar_model.get(Mhalo=srcmass_msun)

        # get escaping fraction
        if self.fesc_kind == "constant":
            fesc = self.fesc_model.f0_esc
        elif self.fesc_kind == "power":
            fesc = self.fesc_model.get(Mhalo=srcmass_msun)
        elif self.fesc_kind == "power_obs":
            # here the escaping fraction is fitted to data that uses stellar mass
            fesc = self.fesc_model.get(Mhalo=fstar * srcmass_msun)
        elif self.fesc_kind == "Gelli2024":
            # mean quantities
            mean_fstar = self.fstar_model.stellar_to_halo_fraction(Mhalo=srcmass_msun)
            mean_Muv = self.fstar_model.UV_magnitude(
                fstar=mean_fstar, mdot=srcmass_msun / ts
            )

            # absolute magnitude with scatter
            Muv = self.fstar_model.UV_magnitude(fstar=fstar, mdot=srcmass_msun / ts)

            # magnitude dependent escaping fraction
            fesc = self.fesc_model.get(delta_Muv=mean_Muv - Muv)
        elif self.fesc_kind == "thesan":
            fesc = self.fesc_model.get(Mhalo=srcmass_msun, z=z)

        # get for star formation history
        nr_switchon: int
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
            srcpos_mpc, srcmass_msun = srcpos_mpc[burst_mask], srcmass_msun[burst_mask]
            if self.fesc_kind == "constant":
                fstar = fstar[burst_mask]
            else:
                fstar, fesc = fstar[burst_mask], fesc[burst_mask]
        else:
            # no bursty model
            nr_switchon = srcmass_msun.size
            self.perc_switchon = 100.0
            pass

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

                # q_ion scenarios need a gridded stellar mass, which the SPICE
                # (SFR-based) branch does not provide.
                self.src_mstar = None

                # normalize flux
                assert self.sources_params.Nion is not None
                normflux = (
                    c.msun2g * self.sources_params.Nion * sfr / (c.m_p * S_star_ref)
                )
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

                # stash the gridded stellar mass (fesc already folded in) for
                # the q_ion normalisation scenarios
                self.src_mstar = srcmstar

                # normalize flux
                assert self.sources_params.Nion is not None
                normflux = (
                    c.msun2g
                    * self.sources_params.Nion
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
            self.src_mstar = np.array((0.0,), dtype=np.float64)
            return np.array((3, 0), dtype=np.int32), np.array((0,), dtype=np.float64)

    def read_haloes(
        self, halo_file: PathType, box_len: float
    ) -> tuple[np.ndarray, np.ndarray]:
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
            # FIXME: unknown attribute
            h = self.h  # type: ignore
            srcmass_msun = hl.get(var="m") / h  # Msun
            srcpos_mpc = hl.get(var="pos") / h  # Mpc
        elif suffix == ".txt":
            # Read haloes from a PKDGrav "halo" txt: positions already in [0, boxsize] Mpc/h.
            hl = np.loadtxt(halo_file)
            srcmass_msun = hl[:, 0] / self.cosmology.h  # Msun
            srcpos_mpc = hl[:, 1:]  # Mpc/h, no centering offset

            # periodic boundary wrap (not reflection)
            srcpos_mpc[srcpos_mpc < 0.0] = self.boxsize + srcpos_mpc[srcpos_mpc < 0.0]
            srcpos_mpc[srcpos_mpc > self.boxsize] = (
                srcpos_mpc[srcpos_mpc > self.boxsize] - self.boxsize
            )

            assert srcpos_mpc.min() >= 0.0
            assert srcpos_mpc.max() <= self.boxsize
            srcpos_mpc /= self.cosmology.h  # Mpc
        return srcpos_mpc, srcmass_msun

    def read_density(self, fbase, z=None):
        """Read coarser density field from C2Ray-formatted file.

        Handles both numpy (.npy) overdensity fields and raw PKDGRAV3 binary.
        """
        file = self.density_basename + fbase
        if file.endswith("npy"):
            overd = np.load(file) - 1.0
        else:
            rdr = t2c.Pkdgrav3data(self.boxsize, self.N, Omega_m=self.cosmology.Om0)
            overd = rdr.load_density_field(file)

        self.ndens = (
            self.cosmology.critical_density0.cgs.value
            * self.cosmology.Ob0
            * (1.0 + overd)
            / (self.mean_molecular * c.m_p)
            * (1 + z) ** 3
        )

        # floor density to avoid zero-valued cells
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

    def run_metallicity_evolution(
        self,
        i_start: int,
        i_end: int,
        time: u.Quantity | None = 10 * u.Myr,
    ) -> None:
        """Run the metallicity evolution pipeline using parameters from YAML.

        Parameters
        ----------
        i_start : Starting snapshot index
        i_end : Ending snapshot index (exclusive)
        time : Time evolved per snapshot (default 10 Myr)
        """
        if self.metallicity_params is None:
            raise ValueError("MetallicityEvolution section required in YAML to run evolution")
        if self.bpass_params is None:
            raise ValueError("BPASSSource section required in YAML to run metallicity evolution")

        paths = _make_metallicity_paths_from_params(
            sim_root=Path(self.metallicity_params.sim_root),
            sim_subdir=self.metallicity_params.sim_subdir,
            halo_subdir=self.metallicity_params.halo_subdir,
            overdensity_subdir=self.metallicity_params.overdensity_subdir,
            halo_filename_fmt=self.metallicity_params.halo_filename_fmt,
            overdensity_filename_fmt=self.metallicity_params.overdensity_filename_fmt,
            output_dir=self.metallicity_params.output_dir,
        )

        cosmo = CosmologyLite(
            h=self.cosmology.h,
            Omega_b=self.cosmology.Ob0,
            Omega_m = self.cosmology.Om0,
            L_box=self.boxsize * u.Mpc / cu.littleh,
        )
        bpass = BPASSYieldTable(Path(self.bpass_params.bpass_dir))

        try:
            first_overd = np.load(paths.overdensity_file(i_start), mmap_mode="r")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Overdensity file for i_start={i_start} not found: {paths.overdensity_file(i_start)}"
            ) from exc

        if first_overd.ndim != 3 or len(set(first_overd.shape)) != 1:
            raise ValueError(f"Overdensity grid must be cubic 3D. Got shape {first_overd.shape}")

        detected_n_cell = int(first_overd.shape[0])
        configured_n_cell = int(self.metallicity_params.N_cell)
        if configured_n_cell != detected_n_cell:
            logger.warning(
                "MetallicityEvolution.N_cell=%d does not match overdensity grid size=%d at i_start=%d. Using %d.",
                configured_n_cell, detected_n_cell, i_start, detected_n_cell,
            )

        sim = MetallicityEvolution(
            cosmology=cosmo,
            paths=paths,
            bpass=bpass,
            stellar_mass_model=StellarMassModelLite(),
            N_cell=detected_n_cell,
            Z_min=self.metallicity_params.Z_min,
        )

        if time is None:
            time = 10 * u.Myr

        logger.info("Starting metallicity evolution from snapshot %d to %d", i_start, i_end)
        sim.run(i_start=i_start, i_end=i_end, time=time)
        logger.info("Metallicity evolution complete. Results saved to %s", paths.output_dir)

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the f_star case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter"""
        self.zred_density = np.loadtxt(self.density_basename + "redshift_density.txt")
        self.zred_sources = np.loadtxt(self.sources_basename + "redshift_sources.txt")
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
        if self.resume:
            # get fields at the resuming redshift
            self.ndens = self.read_density(
                fbase="CDM_200Mpc_2048.%05d.den.256.0" % self.resume, z=self.prev_zdens
            )

            # get extension of the output file
            ext = get_extension_in_folder(path=self.results_basename)
            if ext == ".dat":
                fname = "%sxfrac_z%.3f.dat" % (self.results_basename, self.zred)
                self.xh = t2c.read_cbin(filename=fname, bits=64, order="F")
                self.phi_ion = t2c.read_cbin(
                    filename="%sIonRates_z%.3f.dat"
                    % (self.results_basename, self.zred),
                    bits=32,
                    order="F",
                )
            elif ext == ".npy":
                fname = "%sxfrac_z%.3f.npy" % (self.results_basename, self.zred)
                self.xh = np.load(fname)
                self.phi_ion = np.load(
                    "%sIonRates_z%.3f.npy" % (self.results_basename, self.zred)
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
            self.temp = np.full(self.shape, self.material_params.temp0, order="F")
        else:
            super()._material_init()

    @property
    def fstar_kind(self) -> str:
        return self.sources_params.fstar_kind

    @property
    def acc_kind(self) -> str:
        return self.sources_params.accretion_model

    @property
    def bursty_sfr(self) -> str:
        return self.sources_params.bursty_sfr

    @property
    def fesc_kind(self) -> str:
        return self.sources_params.fesc_model

    def _sources_init(self):
        """Initialize settings to read source files"""
        # --- Stellar-to-Halo Source model ---

        # dictionary with all the f_star parameters
        fstar_pars = {
            "Nion": self.sources_params.Nion,
            "f0": self.sources_params.f0,
            "Mt": self.sources_params.Mt,
            "Mp": self.sources_params.Mp,
            "g1": self.sources_params.g1,
            "g2": self.sources_params.g2,
            "g3": self.sources_params.g3,
            "g4": self.sources_params.g4,
            "alpha_h": self.sources_params.alpha_h,
            "a_s": self.sources_params.a_s,
            "b_s": self.sources_params.b_s,
        }

        # print message that inform of the f_star model employed
        if self.fstar_kind == "fgamma":
            logger.info(
                f"Using constant stellar-to-halo relation model with f_star = {self.sources_params.f0:.1f}, "
                f"Nion = {self.sources_params.Nion:.1f}"
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
        self.fstar_model = StellarToHaloRelation(
            model=self.fstar_kind, pars=fstar_pars, cosmo=self.cosmology
        )

        # --- Halo Accretion Model ---
        # TODO: Create class etc...
        logger.info(f"Using {self.acc_kind} accretion to model.")

        # dictionary with all the burstiness parameters
        if self.bursty_sfr == "instant" or self.bursty_sfr == "integrate":
            bursty_pars = {
                "beta1": self.sources_params.beta1,
                "beta2": self.sources_params.beta2,
                "tB0": self.sources_params.tB0,
                "tQ_frac": self.sources_params.tQ_frac,
                "z0": self.sources_params.z0,
            }

            logger.info(
                f"Using {self.bursty_sfr} bustiness to model the star formation history with parameters: {bursty_pars}."
            )

            # define the burstiness SF model class
            self.bursty_model = BurstySFR(
                model=self.bursty_sfr,
                pars=bursty_pars,
                alpha_h=self.sources_params.alpha_h,
                cosmo=self.cosmology,
            )
        else:
            logger.info("No bustiness model for the star formation history.")

        # --- Escaping fraction Model ---
        fesc_pars = {
            "f0_esc": self.sources_params.f0_esc,
            "Mp_esc": self.sources_params.Mp_esc,
            "al_esc": self.sources_params.al_esc,
        }
        if self.fesc_kind == "constant":
            logger.info(
                "Using constant escaping fraction model with f0_esc = %.1f",
                self.sources_params.f0_esc,
            )
        elif self.fesc_kind == "power":
            logger.info(
                f"Using mass-dependent power law model for the escaping fraction with parameters: {fesc_pars}"
            )
        elif self.fesc_kind == "Gelli2024":
            logger.info(
                f"Using UV magnitude-dependent power law model for the escaping fraction with parameters: {fesc_pars}"
            )

        self.fesc_model = EscapeFraction(model=self.fesc_kind, pars=fesc_pars)

    def _radiation_init(self):
        """Standard radiation init, plus a BPASS photoionization table for every
        metallicity bin. Per slice, the single active table is swapped to the
        volume-average source metallicity (set_radiation_to_metallicity)."""
        super()._radiation_init()
        if self.bpass_params is None:
            raise ValueError("C2Ray_Metals needs a BPASSSource section to build metallicity tables")

        ion_freq_HI = c.ev2fr * self.eth0
        ion_freq_HeII = c.ev2fr * self.ethe1

        scenario = self.bpass_params.norm_scenario
        table_kwargs = dict(
            bpass_dir=self.bpass_params.bpass_dir,
            tau=self.tau,
            freq_min=ion_freq_HI,
            freq_max=10 * ion_freq_HeII,
            S_star_ref=1e48,
            grey=self.grey,
            freq0=ion_freq_HI,
            pl_index=self.cs_pl_idx_h,
            age=self.bpass_params.age,
        )
        if scenario == "bb_qion":
            # Option 1: fitted-Teff black-body SHAPE per Z bin
            self.ztables = BBFittedPhotoTableSet(**table_kwargs)
            logger.info(
                "Built fitted-Teff black-body tables for %d metallicity bins: Z=%s, Teff=%s K",
                self.ztables.n_bins, self.ztables.Z_bin_centers,
                np.array2string(self.ztables.Teff, precision=0),
            )
        else:
            # 'fixed_nion' and 'bpass_qion': BPASS SHAPE per Z bin
            self.ztables = BPASSPhotoTableSet(**table_kwargs)
            logger.info(
                "Built BPASS photoionization tables for %d metallicity bins: %s",
                self.ztables.n_bins, self.ztables.Z_bin_centers,
            )

        if scenario in ("bpass_qion", "bb_qion"):
            # q_ion(Z, age) grid: carries the metallicity- and age-dependent
            # AMPLITUDE that _normalize_sed strips from the tables.
            self.qion_grid = BPASSQionGrid(
                bpass_dir=self.bpass_params.bpass_dir,
                freq_min=ion_freq_HI,
                freq_max=10 * ion_freq_HeII,
            )
            logger.info(
                "Built q_ion(Z, age) grid (scenario '%s'): source amplitudes come from "
                "BPASS ionizing efficiency, Nion is ignored.", scenario,
            )

        self.yield_table = BPASSYieldTable(self.bpass_params.bpass_dir)
        logger.info("Built BPASS total mass-return table for per-sub-step stellar-mass loss.")
        
    def metallicity_output_file(self, i: int):
        """Path to the per-halo metallicity file for snapshot i, written by
        run_metallicity_evolution."""
        if self.metallicity_params is None:
            raise ValueError("MetallicityEvolution section required to locate metallicity files")
        out_dir = (
                Path(self.metallicity_params.output_dir)
                if self.metallicity_params.output_dir else Path(".")
            )
        return out_dir / f"snapshot_{int(i):05d}.metallicities.npy"

    def average_source_metallicity(self, i: int, weight: str = "none") -> float:
        """Volume-average metallicity of all source halos in snapshot i.

        weight : 'none' for a plain mean over halos (matches "average of all
            sources"); 'mstar' for a stellar-mass-weighted mean.
        """
        halos = np.load(self.metallicity_output_file(i), allow_pickle=True)
        fz = halos["f_Z_halo"]
        if weight == "mstar":
            w = halos["M_star"]
            if w.sum() > 0:
                return float(np.average(fz, weights=w))
        return float(np.mean(fz))

    def set_radiation_to_metallicity(self, mean_Z: float) -> float:
        """Set the single active photoionization table to a log-Z interpolation
        between the two BPASS bins bracketing mean_Z and (on GPU) copy it to the
        device. Call once per slice, before raytracing. Returns the (clamped)
        metallicity actually used."""
        thin, thick = self.ztables.get_photo_tables_interp(mean_Z)
        self.photo_thin_table = np.ascontiguousarray(thin)
        self.photo_thick_table = np.ascontiguousarray(thick)
        if self.gpu:
            photo_table_to_device(self.photo_thin_table, self.photo_thick_table)
        lo, hi, w = self.ztables._interp_indices_weight(mean_Z)
        Z_used = float(np.clip(mean_Z, self.ztables.Z_bin_centers[0], self.ztables.Z_bin_centers[-1]))
        logger.info(
            "Slice radiation: mean source Z=%.3e -> interpolated between BPASS bins "
            "Z=%.5f and Z=%.5f (w=%.3f)",
            mean_Z, self.ztables.Z_bin_centers[lo], self.ztables.Z_bin_centers[hi], w,
        )
        return Z_used

    def substep_normflux(self, normflux_birth, mean_Z: float, age_seconds: float):
        """Scale the slice's birth-mass flux to a sub-step by the remaining stellar
        mass fraction from BPASS (winds + SN mass return) at population age
        age_seconds. Slice-mean metallicity mean_Z selects the BPASS Z bin."""
        age_myr = age_seconds / (1e6 * c.year2s)
        frac = self.yield_table.remaining_stellar_fraction(mean_Z, age_myr)
        return normflux_birth * frac

    def scenario_substep_normflux(
        self, normflux_birth, mean_Z: float, age_lo_seconds: float, age_hi_seconds: float
    ):
        """Sub-step source amplitudes under the configured normalisation scenario
        (BPASSSource.norm_scenario in the parameter file). The burst is born at
        slice start (age 0), so sub-step t spans population ages
        [age_lo_seconds, age_hi_seconds] = [t*dt, (t+1)*dt].

        'fixed_nion' : legacy behaviour — Nion-based birth flux scaled by the
                       BPASS remaining-stellar-mass fraction at the sub-step
                       mid-point age.
        'bpass_qion' / 'bb_qion' : amplitude rebuilt from the BPASS ionizing
                       efficiency, interval-averaged over THIS sub-step's age
                       window,

                           normflux = M_star * <q_ion>_[a,b] / S_star_ref,
                           <q_ion>_[a,b] = 1/(b-a) integral_a^b q_ion(Z, age) d(age)

                       with [a, b] = [age_lo, age_hi]. This resolves the steep
                       decline of ionizing output with age across the sub-steps
                       while keeping normflux a RATE: the 1/(b-a) is essential —
                       without it normflux would be a photon count and evolve3D's
                       own *dt (in the chemistry ODE) would double-count it.
                       <q_ion> already contains the population fading, so NO extra
                       remaining-mass factor is applied.
        """
        scenario = self.bpass_params.norm_scenario
        if scenario == "fixed_nion":
            age_mid = 0.5 * (age_lo_seconds + age_hi_seconds)
            return self.substep_normflux(normflux_birth, mean_Z, age_mid)

        if self.src_mstar is None:
            raise RuntimeError(
                "q_ion scenarios need the gridded stellar mass from ionizing_flux "
                "(not available: either ionizing_flux was not called, or a "
                "SFR-based 'spice' fstar model is in use)."
            )
        qbar = self.qion_grid.mean_qion_interval(
            mean_Z, age_lo_seconds / c.year2s, age_hi_seconds / c.year2s
        )
        return self.src_mstar * qbar / 1e48

# ---------------------------------------------------------------------------
# Lightweight metallicity-evolution helper classes (integrated from draft)
# ---------------------------------------------------------------------------

@dataclass
class CosmologyLite:
    h: float
    Omega_b: float
    Omega_m: float
    L_box: u.Quantity = 100 * u.Mpc / cu.littleh

    H_0: u.Quantity = field(init=False)
    rho_c: u.Quantity = field(init=False)

    def __post_init__(self) -> None:
        self.H_0 = 100 * self.h * u.km / u.s / u.Mpc
        rho_c_phys = (3 * self.H_0 ** 2 / (8 * np.pi * cst.G)).to(u.Msun / u.Mpc ** 3)
        self.rho_c = (rho_c_phys / self.h ** 2) * cu.littleh ** 2


@dataclass
class StellarMassModelLite:
    f_star_zero: float = 0.02
    alpha_star: float = -0.3
    pivot_mass: float = 1e10

    def __call__(self, halo_mass_msunh: np.ndarray, h: float) -> np.ndarray:
        halo_mass_msun = halo_mass_msunh * h
        f_star = self.f_star_zero * (halo_mass_msun / self.pivot_mass) ** self.alpha_star
        return f_star * halo_mass_msunh


def flattened_cell_index(
    halo_x: np.ndarray,
    halo_y: np.ndarray,
    halo_z: np.ndarray,
    N_cell: int,
    L_box: u.Quantity,
) -> np.ndarray:
    L = L_box.to_value(u.Mpc / cu.littleh)
    dx = L / N_cell

    i = np.floor(np.asarray(halo_x) / dx).astype(int)
    j = np.floor(np.asarray(halo_y) / dx).astype(int)
    k = np.floor(np.asarray(halo_z) / dx).astype(int)

    i = np.clip(i, 0, N_cell - 1)
    j = np.clip(j, 0, N_cell - 1)
    k = np.clip(k, 0, N_cell - 1)

    return i * N_cell ** 2 + j * N_cell + k


def unflatten_cell_index(idx: np.ndarray, N_cell: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = idx // N_cell ** 2
    j = (idx % N_cell ** 2) // N_cell
    k = idx % N_cell
    return i, j, k


class BPASSYieldTable:
    def __init__(
        self,
        bpass_root: Path,
        n_lifetime_bins: int = 11,
        t_min_myr: float = 0.0,
        t_max_myr: float = 10.0,
        n_time_points: int = 1000,
        bpass_population_mass: float = 1e6,
    ) -> None:
        self.bpass_root = Path(bpass_root)
        self.n_lifetime_bins = n_lifetime_bins
        self.t_min_myr = t_min_myr
        self.t_max_myr = t_max_myr
        self.n_time_points = n_time_points
        self.M_BPASS = bpass_population_mass

        self.metals = np.loadtxt(self.bpass_root / "metals.txt")
        self.lifetime = (10 ** np.loadtxt(self.bpass_root / "ages.txt") * u.yr).to(u.Myr)
        self.lifetime_truncated = self.lifetime[:n_lifetime_bins]

        # build a metals-only and total ejected mass table
        self.Z_list, self.t_grid, self.F_table = self._build_table("metals")
        _, _, self.F_return_table = self._build_table("total")

    def _load_yields(self, metallicity: float) -> pd.DataFrame:
        i_metal = f"{metallicity:.5f}"
        path = self.bpass_root / f"yields-bin-imf135_300.z{i_metal[2:]}.dat"
        return pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            names=["log_age", "H_sw", "He_sw", "Z_sw", "E_sw", "E_sn", "H_sn", "He_sn", "Z_sn"],
        )

    def _build_table(self, channel: str = "metals") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z_list = np.array(sorted(self.metals))
        t_grid = np.linspace(self.t_min_myr, self.t_max_myr, self.n_time_points)
        F_table = np.zeros((len(Z_list), self.n_time_points))

        time_vals_myr = self.lifetime_truncated.to_value(u.Myr)
        time_vals_yr = self.lifetime_truncated.to_value(u.yr)

        for z_idx, Z in enumerate(Z_list):
            yields = self._load_yields(Z)
            if channel == "metals":
                per_bin = (yields["Z_sw"] + yields["Z_sn"]).values[: self.n_lifetime_bins]
            elif channel == "total":
                per_bin = (
                    yields["H_sw"] + yields["He_sw"] + yields["Z_sw"]
                    + yields["H_sn"] + yields["He_sn"] + yields["Z_sn"]
                ).values[: self.n_lifetime_bins]
            else:
                raise ValueError(f"Unknown yield channel: {channel}")

            cumulative = cumulative_trapezoid(per_bin, time_vals_yr, initial=0)
            cumulative /= self.M_BPASS
            F_table[z_idx, :] = np.interp(t_grid, time_vals_myr, cumulative)

        return Z_list, t_grid, F_table

    def time_index(self, t_myr: float) -> int:
        return int(np.searchsorted(self.t_grid, t_myr).clip(0, len(self.t_grid) - 1))

    def _z_weights(self, metallicity):
        """Vectorized bracketing indices and log-Z weights, clamped to the Z range."""
        Z = np.clip(np.asarray(metallicity, dtype=float), self.Z_list[0], self.Z_list[-1])
        hi = np.searchsorted(self.Z_list, Z).clip(1, len(self.Z_list) - 1)
        lo = hi - 1
        w = (np.log10(Z) - np.log10(self.Z_list[lo])) / (
            np.log10(self.Z_list[hi]) - np.log10(self.Z_list[lo])
        )
        return lo, hi, np.clip(w, 0.0, 1.0)

    def metal_produced(self, metallicity: np.ndarray, stellar_mass: np.ndarray, t1_idx: int, t2_idx: int) -> np.ndarray:
        lo, hi, w = self._z_weights(metallicity)
        dF_lo = self.F_table[lo, t2_idx] - self.F_table[lo, t1_idx]
        dF_hi = self.F_table[hi, t2_idx] - self.F_table[hi, t1_idx]
        return ((1.0 - w) * dF_lo + w * dF_hi) * stellar_mass

    def mass_returned(self, metallicity: np.ndarray, stellar_mass: np.ndarray, t1_idx: int, t2_idx: int) -> np.ndarray:
        """Total (H + He + Z, winds + SN) mass ejected between the two time
        indices, log-Z-interpolated between the bracketing BPASS bins."""
        lo, hi, w = self._z_weights(metallicity)
        dF_lo = self.F_return_table[lo, t2_idx] - self.F_return_table[lo, t1_idx]
        dF_hi = self.F_return_table[hi, t2_idx] - self.F_return_table[hi, t1_idx]
        return ((1.0 - w) * dF_lo + w * dF_hi) * stellar_mass

    def remaining_stellar_fraction(self, metallicity: float, t_myr: float) -> float:
        """Fraction of the initial stellar mass still locked in stars at population
        age t_myr, i.e. 1 - cumulative BPASS total (wind+SN) mass-return fraction,
        log-Z-interpolated between the two bracketing BPASS bins."""
        lo, hi, w = self._z_weights(metallicity)
        t_idx = self.time_index(t_myr)
        F = (1.0 - w) * self.F_return_table[lo, t_idx] + w * self.F_return_table[hi, t_idx]
        return float(1.0 - F)


class MetallicityEvolution:
    HALO_TXT_COLUMNS = ["Mhalo", "x", "y", "z"]

    def __init__(
        self,
        cosmology: CosmologyLite,
        paths: Optional[object] = None,
        bpass: Optional[BPASSYieldTable] = None,
        stellar_mass_model: StellarMassModelLite = StellarMassModelLite(),
        N_cell: int = 100,
        Z_min: float = 1e-5,
    ) -> None:
        self.cosmology = cosmology
        self.paths = paths
        self.bpass = bpass if bpass is not None else BPASSYieldTable(Path("."))
        self.stellar_mass_model = stellar_mass_model
        self.N_cell = N_cell
        self.Z_min = Z_min
        self.M_Z_cell: np.ndarray = np.zeros(self.N_cell ** 3)

    @property
    def N_tot(self) -> int:
        return self.N_cell ** 3

    def stellar_mass(self, halo_mass: np.ndarray) -> np.ndarray:
        return self.stellar_mass_model(halo_mass, self.cosmology.h)

    def cell_index(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return flattened_cell_index(x, y, z, self.N_cell, self.cosmology.L_box)

    def load_halo_txt(self, i: int) -> np.ndarray:
        cols = self.HALO_TXT_COLUMNS
        df = pd.read_csv(
            self.paths.halo_file(i),
            sep=r"\s+", comment="#", header=None, names=cols, dtype="float64",
        )
        return df.to_records(index=False)   # structured array, same field names
    
    def load_overdensity(self, i: int) -> np.ndarray:
        return np.load(self.paths.overdensity_file(i))

    def load_overdensity_safe(self, i: int) -> Optional[np.ndarray]:
        try:
            return self.load_overdensity(i)
        except FileNotFoundError:
            logger.warning("Missing overdensity file for snapshot %d", i)
            return None

    def initialize_halos(self, i: int, i_start: int) -> np.ndarray:
        halo_positions = self.load_halo_txt(i)

        halos = rfn.append_fields(
            halo_positions,
            names=["cell_id", "M_star", "M_delta_Z", "f_Z_halo"],
            data=[
                np.zeros(len(halo_positions), dtype=int),
                np.zeros(len(halo_positions)),
                np.zeros(len(halo_positions)),
                np.zeros(len(halo_positions)),
            ],
            usemask=False,
        )

        halos["cell_id"] = self.cell_index(halos["x"], halos["y"], halos["z"])
        halos["M_star"] = self.stellar_mass(halos["Mhalo"])

        if i == i_start:
            halos["f_Z_halo"][:] = self.Z_min
        else:
            prev = np.load(self.paths.gridded_file(i - 1), allow_pickle=True)
            halos["f_Z_halo"] = prev["f_Z_cell"][halos["cell_id"]]

        return halos

    def evolve(self, halos: np.ndarray, i: int, t1_idx: int, t2_idx: int, save: bool = True) -> np.ndarray:
        out = halos.copy()
        # Metals deposited into the gas: metals-only (Z winds + SN) channel.
        delta_Z = self.bpass.metal_produced(out["f_Z_halo"], out["M_star"], t1_idx=t1_idx, t2_idx=t2_idx)
        # Mass leaving the stars: FULL H + He + Z (winds + SN) ejecta. The
        # returned H/He rejoins gas whose mass is already set by the
        # overdensity grid, so only delta_Z enters the metal budget.
        delta_ej = self.bpass.mass_returned(out["f_Z_halo"], out["M_star"], t1_idx=t1_idx, t2_idx=t2_idx)
        out["M_delta_Z"] = delta_Z
        out["M_star"] -= delta_ej

        if save:
            np.save(self.paths.metallicity_file(i), out)
        return out

    def baryonic_mass_per_cell(self, overdensity: np.ndarray) -> np.ndarray:
        ngrid = overdensity.shape[0]
        cell_size = self.cosmology.L_box / ngrid
        cell_volume = cell_size ** 3
        M_b = self.cosmology.rho_c * (self.cosmology.Omega_b / self.cosmology.Omega_m) * overdensity * cell_volume
        return M_b.value.flatten()

    def accumulate_metals(self, evolved: np.ndarray) -> None:
        self.M_Z_cell += np.bincount(evolved["cell_id"], weights=evolved["M_delta_Z"], minlength=self.N_tot)

    def grid_snapshot(self, i: int, evolved: np.ndarray, overdensity: np.ndarray, save: bool = True) -> np.ndarray:
        N_tot = self.N_tot
        cell_idx = evolved["cell_id"].astype(int)

        sum_halo = np.bincount(cell_idx, weights=evolved["Mhalo"], minlength=N_tot)
        sum_star = np.bincount(cell_idx, weights=evolved["M_star"], minlength=N_tot)

        M_b_cell = self.baryonic_mass_per_cell(overdensity)
        if M_b_cell.size != self.M_Z_cell.size:
            raise ValueError(
                f"Grid-size mismatch at snapshot {i}: metal grid has {self.M_Z_cell.size} cells "
                f"(N_cell={self.N_cell}), overdensity has {M_b_cell.size} cells "
                f"(shape={overdensity.shape})."
            )
        # Gas is pre-enriched to the metallicity floor Z_min, so each cell's
        # metal budget is the floor contribution (Z_min * M_b) plus the metals
        # produced by stars (self.M_Z_cell). The resulting metal fraction is
        #   f_Z = (Z_min * M_b + M_Z_produced) / M_b = Z_min + M_Z_produced / M_b
        # which is guaranteed >= Z_min wherever there is gas and only increases
        # as stars enrich it. Cells with no baryons (M_b == 0) have no defined
        # metallicity and are left at 0.
        f_Z_cell = np.where(M_b_cell > 0, self.Z_min + self.M_Z_cell / M_b_cell, 0.0)

        gridded = np.zeros(
            N_tot,
            dtype=[
                ("cell_index", int),
                ("M_halo_cell", float),
                ("M_star_cell", float),
                ("M_Z_cell", float),
                ("f_Z_cell", float),
            ],
        )
        gridded["cell_index"] = np.arange(N_tot)
        gridded["M_halo_cell"] = sum_halo
        gridded["M_star_cell"] = sum_star
        gridded["M_Z_cell"] = self.M_Z_cell
        gridded["f_Z_cell"] = f_Z_cell

        if save:
            np.save(self.paths.gridded_file(i), gridded)
        return gridded

    def run(self, i_start: int, i_end: int, time: u.Quantity = 10 * u.Myr, prefetch_workers: int = 2) -> None:
        total_time_myr = time.to_value(u.Myr) if isinstance(time, u.Quantity) else float(time)
        t1_idx = self.bpass.time_index(0.0)
        t2_idx = self.bpass.time_index(total_time_myr)

        with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
            current_overd = self.load_overdensity_safe(i_start)

            for i in tqdm(range(i_start, i_end), desc="Snapshots"):
                if current_overd is None:
                    if i + 1 < i_end:
                        current_overd = self.load_overdensity_safe(i + 1)
                    continue

                if (
                    current_overd.ndim != 3
                    or current_overd.shape[0] != self.N_cell
                    or current_overd.shape[1] != self.N_cell
                    or current_overd.shape[2] != self.N_cell
                ):
                    raise ValueError(
                        f"Overdensity shape mismatch at snapshot {i}: expected "
                        f"({self.N_cell}, {self.N_cell}, {self.N_cell}), got {current_overd.shape}."
                    )

                next_future = pool.submit(self.load_overdensity_safe, i + 1) if i + 1 < i_end else None

                halos = self.initialize_halos(i, i_start)
                evolved = self.evolve(halos, i, t1_idx=t1_idx, t2_idx=t2_idx)
                self.accumulate_metals(evolved)
                self.grid_snapshot(i, evolved, current_overd)

                if next_future is not None:
                    current_overd = next_future.result()


def _make_metallicity_paths(sim_root: Path, sim_subdir: str) -> object:
    class P:
        def __init__(self, sim_root, sim_subdir):
            self.sim_root = Path(sim_root)
            self.sim_subdir = sim_subdir
            self.halo_subdir = "sources/"
            self.overdensity_subdir = "grids/nc100/"

        def halo_file(self, i: int) -> Path:
            return self.sim_root / self.sim_subdir / self.halo_subdir / f"{self.sim_subdir}.{i:05d}.halo.txt"

        def overdensity_file(self, i: int) -> Path:
            return self.sim_root / self.sim_subdir / self.overdensity_subdir / f"{self.sim_subdir}.{i:05d}.overden.npy"

        def metallicity_file(self, i: int) -> Path:
            return Path(".") / f"snapshot_{i:05d}.metallicities.npy"

        def gridded_file(self, i: int) -> Path:
            return Path(".") / f"snapshot_{i:05d}.gridded_halos.npy"

    return P(sim_root, sim_subdir)


def _make_metallicity_paths_from_params(
    sim_root: Path,
    sim_subdir: str,
    halo_subdir: str = "sources/",
    overdensity_subdir: str = "grids/nc100/",
    halo_filename_fmt: str = "{sim_subdir}.{i:05d}.halo.txt",
    overdensity_filename_fmt: str = "{sim_subdir}.{i:05d}.overden.npy",
    output_dir: Optional[str] = None,
) -> object:
    """Create a paths object using YAML-configured file format templates."""

    class P:
        def __init__(
            self,
            sim_root,
            sim_subdir,
            halo_subdir,
            overdensity_subdir,
            halo_filename_fmt,
            overdensity_filename_fmt,
            output_dir,
        ):
            self.sim_root = Path(sim_root)
            self.sim_subdir = sim_subdir
            self.halo_subdir = halo_subdir
            self.overdensity_subdir = overdensity_subdir
            self.halo_filename_fmt = halo_filename_fmt
            self.overdensity_filename_fmt = overdensity_filename_fmt
            self.output_dir = Path(output_dir) if output_dir else Path(".")

        def halo_file(self, i: int) -> Path:
            fname = self.halo_filename_fmt.format(sim_subdir=self.sim_subdir, i=i)
            return self.sim_root / self.sim_subdir / self.halo_subdir / fname

        def overdensity_file(self, i: int) -> Path:
            fname = self.overdensity_filename_fmt.format(sim_subdir=self.sim_subdir, i=i)
            return self.sim_root / self.sim_subdir / self.overdensity_subdir / fname

        def metallicity_file(self, i: int) -> Path:
            return self.output_dir / f"snapshot_{i:05d}.metallicities.npy"

        def gridded_file(self, i: int) -> Path:
            return self.output_dir / f"snapshot_{i:05d}.gridded_halos.npy"

    return P(
        sim_root,
        sim_subdir,
        halo_subdir,
        overdensity_subdir,
        halo_filename_fmt,
        overdensity_filename_fmt,
        output_dir,
    )


def run_metallicity_from_c2ray(
    c2ray: C2Ray,
    sim_root: Path,
    sim_subdir: str,
    i_start: int,
    i_end: int,
    bpass_root: Path,
    N_cell: int = 100,
):
    cosmo = CosmologyLite(h=c2ray.cosmology.h,
                          Omega_b=c2ray.cosmology.Ob0,
                          Omega_m=c2ray.cosmology.Om0,
                          L_box=c2ray.boxsize * u.Mpc / cu.littleh)
    paths = _make_metallicity_paths(sim_root, sim_subdir)
    bpass = BPASSYieldTable(bpass_root)
    sim = MetallicityEvolution(cosmology=cosmo, paths=paths, bpass=bpass, N_cell=N_cell)
    sim.run(i_start=i_start, i_end=i_end)
