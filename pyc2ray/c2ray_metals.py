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
from pyc2ray.evolve import evolve3D_multigroup
from pyc2ray.radiation import BlackBodySource, BPASSSource

# Additional imports for metallicity evolution
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from numpy.lib import recfunctions as rfn
from scipy.integrate import cumulative_trapezoid, trapezoid
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

        # Cell-by-cell state, rebuilt from scratch by prepare_cellwise_slice at
        # every slice. Empty dict == no active cellwise slice; the source count
        # is kept so evolve3D_cellwise can refuse a stale/mismatched mapping
        # instead of silently indexing the wrong sources.
        self._cw_bins: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cw_nsrc: int = 0

        # M_star-weighted metallicity of each binned source cell, built by
        # ionizing_flux on the SAME binning as the flux (so it is registered to
        # the sources by construction, whatever bin_sources' mesh convention is).
        # None when unavailable -> prepare_cellwise_slice falls back to the
        # legacy cell-grid lookup.
        self.src_fz: np.ndarray | None = None



    # =====================================================================================================
    # USER DEFINED METHODS
    # ====================================================================================================

    def ionizing_flux(
        self,
        file: PathType,
        z: float,
        dt: float | None = None,
        save_Mstar: bool = False,
        i: int | None = None,
        dt_slice: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read sources from a C2Ray-formatted file
        Parameters
        ----------
        file : Filename to read.
        z : redshift
        dt : time-step in Myrs.
        save_Mstar : whether to save the stellar mass of the sources (not used)
        i : snapshot index of this slice. Only needed for the cell-by-cell
            metallicity regime: the per-halo metallicities of snapshot i (which
            already carry the i-1 lookback, see below) are binned onto the
            source mesh alongside the stellar mass, giving
            self.src_fz (the M_star-weighted Z of each source cell). Because the
            two binnings share the same call, the metallicity is registered to
            the sources exactly, with no assumption that the source mesh and the
            metallicity mesh use the same cell width. When omitted,
            prepare_cellwise_slice falls back to the legacy cell-grid lookup.
        dt_slice : full length of this redshift slice in seconds (the sub-step dt
            times the number of sub-steps). Required only when
            BPASSSource.sfr_normalised_amplitude is on, where it sets the burst
            fraction min(1, dt_slice / t_form) applied to the gridded stellar
            mass. Ignored otherwise.


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

        # Per-halo metallicity for the cellwise regime, row-aligned with the halo
        # file just read (None if unavailable, e.g. the first slice).
        #
        # Snapshot i, NOT i-1: run_metallicity_evolution writes one row per halo
        # of snapshot i, and sets that halo's f_Z_halo from the GRIDDED field of
        # snapshot i-1. So the i-1 lookback is already applied one level down,
        # and these values are the pre-enrichment metallicity of exactly the
        # haloes in this slice's halo file. Reading i-1 here would compare two
        # different halo lists.
        self.src_fz = None
        fz_halo = (
            self._read_f_Z_halo(i, n_halo=srcmass_msun.size)
            if (self.cellwise and i is not None)
            else None
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
            if fz_halo is not None:
                # keep the per-halo metallicity aligned with the surviving haloes
                fz_halo = fz_halo[burst_mask]
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
                # the q_ion normalisation scenarios, optionally rescaled to the
                # mass FORMED IN THIS SLICE (see sfr_burst_fraction)
                self.src_mstar = srcmstar * self.sfr_burst_fraction(z, dt_slice)

                if fz_halo is not None:
                    # Bin M_star * f_Z on the SAME mesh, then divide: the result
                    # is the M_star-weighted metallicity of each source cell,
                    # aligned row-for-row with srcpos/srcmstar by construction.
                    # Both binnings keep exactly the cells with a positive
                    # weight sum, and f_Z >= Z_min > 0, so the two source lists
                    # are identical.
                    _, srcmstarZ = bin_sources(
                        srcpos_mpc=srcpos_mpc,
                        mstar_msun=mstar_msun * fz_halo,
                        boxsize=self.boxsize / self.cosmology.h,
                        meshsize=self.N + 1,
                    )
                    if srcmstarZ.shape != srcmstar.shape:
                        raise RuntimeError(
                            "metallicity binning did not reproduce the stellar-mass "
                            f"source list ({srcmstarZ.shape} vs {srcmstar.shape}); "
                            "cannot align f_Z with the sources."
                        )
                    self.src_fz = srcmstarZ / srcmstar
                    logger.info(
                        " source-cell metallicity (M_star-weighted): min, mean, max "
                        "= %.3e  %.3e  %.3e",
                        self.src_fz.min(), self.src_fz.mean(), self.src_fz.max(),
                    )

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
            # Properly-shaped empty arrays: srcpos is (0, 3) like the populated
            # branch's bin_sources output, and src_mstar/normflux are length 0
            # so every downstream length check stays consistent.
            self.src_mstar = np.zeros(0, dtype=np.float64)
            return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.float64)

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
            # The PKDGrav3 .npy grids store rho_m / rho_crit,0 (comoving), whose
            # box mean is Omega_m -- NOT 1 + delta, whose mean is 1. Divide by
            # Omega_m before use: the expression below already supplies Omega_b,
            # so feeding it the raw field under-densities the box by a factor
            # Omega_m (~3.1x here), which suppresses recombinations by ~10x and
            # makes reionization finish far too early.
            overd = np.load(file) / self.cosmology.Om0 - 1.0
        else:
            rdr = t2c.Pkdgrav3data(self.boxsize, self.N, Omega_m=self.cosmology.Om0)
            overd = rdr.load_density_field(file)

        # A correctly normalised overdensity field has <1 + delta> = 1. Warn
        # rather than raise: a sub-percent deviation is sample variance, but a
        # factor-of-a-few offset is a unit/convention mismatch in the input.
        mean_1pd = float(np.mean(1.0 + overd))
        if not 0.98 < mean_1pd < 1.02:
            logger.warning(
                "Density field %s has <1+delta> = %.4f, expected 1.0. The gas "
                "density (and hence recombination rate, which goes as n^2) will "
                "be wrong by that factor. Check the file's normalisation "
                "convention: does it store 1+delta, or rho/rho_crit?",
                file, mean_1pd,
            )

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
 <1+delta> = %.4f (must be 1.0)
 min, mean and max density : %.3e  %.3e  %.3e [1/cm3]""",
            file,
            mean_1pd,
            self.ndens.min(),
            self.ndens.mean(),
            self.ndens.max(),
        )

    def _load_snapshot_redshifts(self) -> dict[int, float]:
        """{snapshot index: redshift} from <inputs_basename>redshift_checkpoints.txt
        (two columns: index, redshift) -- the same file the driver uses to build
        its slice loop, so the two agree by construction."""
        path = self.inputs_basename + "redshift_checkpoints.txt"
        try:
            idx, zred = np.loadtxt(path, dtype=float, unpack=True)
        except OSError as exc:
            raise FileNotFoundError(
                f"sfr_normalised_yield needs a snapshot->redshift map and could not "
                f"read {path}. Pass redshifts={{index: z}} to "
                "run_metallicity_evolution instead."
            ) from exc
        return {int(i): float(z) for i, z in zip(np.atleast_1d(idx), np.atleast_1d(zred))}

    def run_metallicity_evolution(
        self,
        i_start: int,
        i_end: int,
        time: u.Quantity | None = 10 * u.Myr,
        redshifts: dict[int, float] | None = None,
    ) -> None:
        """Run the metallicity evolution pipeline using parameters from YAML.

        Parameters
        ----------
        i_start : Starting snapshot index
        i_end : Ending snapshot index (exclusive)
        time : Time evolved per snapshot (default 10 Myr)
        redshifts : optional {snapshot index: redshift} map. Only needed when
            MetallicityEvolution.sfr_normalised_yield is on, to evaluate
            source_lifetime(z) per snapshot. Defaults to reading
            <inputs_basename>redshift_checkpoints.txt (columns: index, z).
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

        # Use the SAME stellar-to-halo relation as ionizing_flux, so a halo's
        # metal yield and its photon output come from one stellar mass.
        stellar_mass_model = make_stellar_mass_model(
            self.sources_params,
            Omega_b=self.cosmology.Ob0,
            Omega_m=self.cosmology.Om0,
        )
        logger.info(
            "Enrichment stellar-mass model: %s (Sources.fstar_kind='%s')",
            type(stellar_mass_model).__name__, self.fstar_kind,
        )

        if time is None:
            time = 10 * u.Myr

        # Burst fraction: what share of a halo's cumulative stellar mass formed
        # during this snapshot. Uses the SAME source_lifetime(z) as
        # C2Ray_Metals.sfr_burst_fraction, so the metal and photon pipelines
        # agree on which stars count as young.
        burst_fraction = None
        if self.metallicity_params.sfr_normalised_yield:
            zmap = redshifts if redshifts is not None else self._load_snapshot_redshifts()
            dt_slice_s = float(time.to_value(u.s))

            def burst_fraction(idx: int, _zmap=zmap, _dt=dt_slice_s) -> float:
                key = int(idx)
                if key not in _zmap:
                    raise KeyError(
                        f"No redshift for snapshot {key}; sfr_normalised_yield needs "
                        "one per snapshot. Pass redshifts={index: z} to "
                        "run_metallicity_evolution, or check "
                        f"{self.inputs_basename}redshift_checkpoints.txt."
                    )
                ts = float(self.fstar_model.source_lifetime(z=_zmap[key]))
                if not np.isfinite(ts) or ts <= 0.0:
                    raise RuntimeError(
                        f"source_lifetime(z={_zmap[key]}) returned {ts}; check "
                        "Sources.alpha_h."
                    )
                return min(1.0, _dt / ts)

            logger.info(
                "SFR-normalised yield ENABLED: metals come from the stellar mass "
                "formed in each snapshot (burst fraction min(1, dt/t_form)), not "
                "the cumulative M_star. Slice length %.3f Myr.",
                time.to_value(u.Myr),
            )
        else:
            logger.info(
                "SFR-normalised yield disabled: each snapshot produces metals "
                "from the FULL cumulative M_star (previous behaviour)."
            )

        sim = MetallicityEvolution(
            cosmology=cosmo,
            paths=paths,
            bpass=bpass,
            stellar_mass_model=stellar_mass_model,
            N_cell=detected_n_cell,
            Z_min=self.metallicity_params.Z_min,
            ism_self_enrichment=self.metallicity_params.ism_self_enrichment,
            metal_retention=self.metallicity_params.metal_retention,
            burst_fraction=burst_fraction,
        )

        if self.metallicity_params.ism_self_enrichment:
            logger.info(
                "ISM self-enrichment ENABLED (metal_retention=%.3f): "
                "f_Z_halo = f_Z_cell + y(Z)*f_ret*M_star/(f_b*M_halo). Lifetime "
                "metal yield y(Z) over the BPASS bins: %s",
                self.metallicity_params.metal_retention,
                np.array2string(bpass.y_total, precision=4),
            )
        else:
            logger.info(
                "ISM self-enrichment disabled: f_Z_halo is the accreted cell "
                "metallicity only (previous behaviour)."
            )

        logger.info("Starting metallicity evolution from snapshot %d to %d", i_start, i_end)
        sim.run(i_start=i_start, i_end=i_end, time=time)
        logger.info("Metallicity evolution complete. Results saved to %s", paths.output_dir)

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the f_star case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter"""
        # These files have TWO columns (snapshot index, redshift). find_bins ->
        # np.digitize needs a 1-D array of bins, so take column 1 -- matching
        # c2ray_fstar._redshift_init. Loading both columns raises
        # "ValueError: object too deep for desired array" on the resume path.
        self.zred_density = np.loadtxt(
            self.density_basename + "redshift_density.txt", usecols=1
        )
        self.zred_sources = np.loadtxt(
            self.sources_basename + "redshift_sources.txt", usecols=1
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

    def _resume_density_fbase(self, i: int) -> str:
        """Filename fragment (appended to density_basename) of the density field
        to resume from, following the same naming convention as the enrichment
        pipeline rather than a hardcoded CubeP3M 200 Mpc name."""
        mp = self.metallicity_params
        if mp is None:
            raise ValueError(
                "Cannot build the density filename to resume from: add a "
                "MetallicityEvolution section (sim_subdir / "
                "overdensity_filename_fmt) to the parameter file, or override "
                "_resume_density_fbase."
            )
        return mp.overdensity_filename_fmt.format(sim_subdir=mp.sim_subdir, i=int(i))

    def _material_init(self):
        """Initialize material properties of the grid"""
        if self.resume:
            # get fields at the resuming redshift. NOTE: read_density assigns
            # self.ndens itself and returns None -- assigning its return value
            # here silently left the density field as None.
            self.read_density(
                fbase=self._resume_density_fbase(self.resume), z=self.prev_zdens
            )

            # results_basename is a Path, so build paths with "/" -- "%s" + name
            # drops the separator and yields e.g. ".../results/06_cellwisexfrac_z9.363.npy".
            # get_extension_in_folder() is avoided for the same reason: it does
            # glob(str(path) + "xfrac*"), which matches nothing and then raises
            # IndexError on arr[0]. Pick the extension by asking which file exists.
            npy = self.results_basename / ("xfrac_z%.3f.npy" % self.zred)
            dat = self.results_basename / ("xfrac_z%.3f.dat" % self.zred)
            if npy.exists():
                fname = npy
                self.xh = np.load(fname)
                self.phi_ion = np.load(
                    self.results_basename / ("IonRates_z%.3f.npy" % self.zred)
                )
            elif dat.exists():
                fname = dat
                self.xh = t2c.read_cbin(filename=str(fname), bits=64, order="F")
                self.phi_ion = t2c.read_cbin(
                    filename=str(
                        self.results_basename / ("IonRates_z%.3f.dat" % self.zred)
                    ),
                    bits=32,
                    order="F",
                )
            else:
                raise FileNotFoundError(
                    f"Resume file not found: {npy} (nor the .dat form). The run's "
                    "output directory must contain the slice being resumed from."
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
    def cellwise(self) -> bool:
        """Whether the cell-by-cell (two-group) metallicity regime is enabled."""
        return bool(self.bpass_params is not None and self.bpass_params.cellwise_metallicity)

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

        if self.cellwise:
            if self.metallicity_params is None:
                raise ValueError(
                    "cellwise_metallicity requires a MetallicityEvolution section "
                    "(it reads per-cell f_Z_cell from the gridded snapshot files)."
                )
            if scenario not in ("bpass_qion", "bb_qion"):
                raise ValueError(
                    "cellwise_metallicity requires a q_ion scenario "
                    "('bpass_qion' or 'bb_qion')."
                )
            logger.info(
                "Cell-by-cell metallicity regime ENABLED (per-bin): per slice, each "
                "source is given its cell metallicity f_Z_cell (new cells at the "
                "Z_min=%.1e floor) and flux-split across the %d BPASS bins; each "
                "occupied bin is raytraced with its own table and bin-center q_ion.",
                self.metallicity_params.Z_min, self.ztables.n_bins,
            )

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

    def average_source_metallicity(self, i: int, weight: str = "mstar") -> float:
        """Average metallicity of the source halos in snapshot i.

        weight : 'mstar' (default) for a stellar-mass-weighted mean -- the
            quantity that matters for the emitted spectrum, since the ionizing
            output tracks M_star; 'none' for a plain mean over halos, which is
            dominated by the numerous small, barely-enriched haloes and biases
            the slice metallicity low.
        """
        halos = np.load(self.metallicity_output_file(i), allow_pickle=True)
        fz = halos["f_Z_halo"]
        if weight == "mstar":
            w = np.asarray(halos["M_star"], dtype=float)
            w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
            if w.sum() > 0:
                return float(np.average(fz, weights=w))
            logger.warning(
                "Snapshot %d has no positive M_star; falling back to an unweighted "
                "mean metallicity.", i,
            )
        return float(np.mean(fz))

    def slice_metallicity(self, i: int) -> float:
        """Effective volume metallicity for slice i, used for the single-table
        (non-cellwise) path. Returns the fixed benchmark value if
        BPASSSource.fixed_metallicity is set (constant Z for the whole run),
        otherwise the per-slice source average (evolving Z).

        The average is stellar-mass weighted: the single active table stands in
        for the whole source population, so it should be the metallicity the
        photons actually come from, not the metallicity of the median halo."""
        if self.bpass_params is not None and self.bpass_params.fixed_metallicity is not None:
            return float(self.bpass_params.fixed_metallicity)
        return self.average_source_metallicity(i, weight="mstar")

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

    def sfr_burst_fraction(self, z: float, dt_slice: float | None) -> float:
        """Fraction of a halo's CUMULATIVE stellar mass that formed during this
        slice: min(1, dt_slice / t_form(z)), with t_form = source_lifetime(z) =
        1 / (alpha_h (1+z) H(z)), the stellar-mass e-folding time of the
        accretion model.

        Returns 1.0 (no rescaling, previous behaviour) unless
        BPASSSource.sfr_normalised_amplitude is enabled.

        Why this exists: q_ion(Z, age) is an ionizing rate per unit mass of a
        COEVAL population of that age, but self.src_mstar is the cumulative
        stellar mass, most of which is far older than 10 Myr and emits no
        ionizing photons. Multiplying them treats a galaxy's entire stellar
        population as newly born, every slice. Scaling by dt_slice / t_form turns
        the amplitude into

            normflux = f_esc * SFR * dt_slice * <q_ion> / S_star_ref,
            SFR = M_star / t_form,

        which is what makes f_esc an escape fraction again rather than a lumped
        calibration constant. The factor is capped at 1: a halo cannot form more
        stars in a slice than it has ever formed.

        NOTE the correction is redshift dependent (t_form grows faster than the
        slice length as z falls), so it changes the SHAPE of the reionization
        history, not just its normalisation -- f_esc cannot absorb it.
        """
        if not (self.bpass_params and self.bpass_params.sfr_normalised_amplitude):
            return 1.0
        if dt_slice is None:
            raise RuntimeError(
                "BPASSSource.sfr_normalised_amplitude needs the slice length: "
                "pass dt_slice=<sub-step dt * number of sub-steps> (in seconds) "
                "to ionizing_flux."
            )
        t_form = float(self.fstar_model.source_lifetime(z=z))
        if not np.isfinite(t_form) or t_form <= 0.0:
            raise RuntimeError(
                f"source_lifetime(z={z}) returned {t_form}; cannot build the "
                "burst fraction. Check Sources.alpha_h."
            )
        frac = min(1.0, float(dt_slice) / t_form)
        logger.info(
            " SFR-normalised amplitude: t_form = %.3f Myr, slice = %.3f Myr, "
            "burst fraction = %.4f (M_star scaled to the mass formed this slice)",
            t_form / (1e6 * c.year2s),
            float(dt_slice) / (1e6 * c.year2s),
            frac,
        )
        return frac

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
        qbar = self._qion_rate_interval(mean_Z, age_lo_seconds, age_hi_seconds)
        return self.src_mstar * qbar / 1e48

    def _qion_rate_interval(self, Z, age_lo_seconds, age_hi_seconds, n_sample=256):
        """Per-INITIAL-solar-mass ionizing rate for a sub-step, interval-averaged
        over the age window [age_lo, age_hi] (seconds).

        Default: <q_ion(Z, age)>_[a,b] (the standard behaviour).

        If BPASSSource.qion_scale_surviving_mass is set, the rate is instead
        <(1 - F_return(Z, age)) * q_ion(Z, age)>_[a,b], i.e. it scales by the
        surviving stellar-mass fraction so that

            photon_rate = M_current(age) * q_ion(age),
            M_current   = M_initial * (1 - F_return(age)).

        WARNING: q_ion is already normalised per INITIAL population mass and
        intrinsically declines as stars die, so this surviving-mass weighting
        double-counts stellar mass loss. It is an opt-in, physically
        non-standard choice; leave the flag off for the standard behaviour.
        """
        a = age_lo_seconds / c.year2s
        b = age_hi_seconds / c.year2s
        use_surv = bool(self.bpass_params and self.bpass_params.qion_scale_surviving_mass)
        if not use_surv:
            return self.qion_grid.mean_qion_interval(Z, a, b)
        if b <= a:
            surv = self.yield_table.remaining_stellar_fraction(Z, a / 1e6)
            return self.qion_grid.qion_at(Z, a) * surv
        ages = np.linspace(a, b, int(n_sample))
        q = np.array([self.qion_grid.qion_at(Z, age) for age in ages])
        surv = np.array(
            [self.yield_table.remaining_stellar_fraction(Z, age / 1e6) for age in ages]
        )
        return float(trapezoid(q * surv, ages) / (b - a))

    # =====================================================================================================
    # CELL-BY-CELL METALLICITY REGIME (two-group)
    # =====================================================================================================

    def gridded_metallicity_file(self, i: int) -> Path:
        """Path to the per-cell gridded metallicity file for snapshot i, written
        by run_metallicity_evolution (holds the f_Z_cell field)."""
        if self.metallicity_params is None:
            raise ValueError("MetallicityEvolution section required to locate gridded files")
        out_dir = (
            Path(self.metallicity_params.output_dir)
            if self.metallicity_params.output_dir else Path(".")
        )
        return out_dir / f"snapshot_{int(i):05d}.gridded_halos.npy"

    def _read_f_Z_halo(self, i: int, n_halo: int):
        """Per-halo f_Z_halo from snapshot i's metallicity file, row-aligned with
        snapshot i's halo file. Returns None if the file is absent (e.g. the
        first slice).

        Pass the CURRENT slice index, not i-1: these rows describe snapshot i's
        haloes, and their f_Z_halo was already taken from snapshot i-1's gridded
        metal field by run_metallicity_evolution. Passing i-1 would line up two
        different halo lists and (correctly) trip the row-count check below.

        The alignment is positional: run_metallicity_evolution writes one row per
        halo in the order of the same halo txt that read_haloes reads, so a row
        count mismatch means the two pipelines are pointed at different halo
        files and the mapping would be meaningless -- that raises rather than
        silently mis-assigning metallicities.
        """
        path = self.metallicity_output_file(i)
        if not path.exists():
            logger.info(
                "Cellwise: no per-halo metallicity file for snapshot %d (%s).", i, path
            )
            return None
        halos = np.load(path, allow_pickle=True)
        if halos.shape[0] != n_halo:
            raise ValueError(
                f"Per-halo metallicity file {path} has {halos.shape[0]} rows but the "
                f"halo file for this slice has {n_halo} haloes. The metallicity "
                "pipeline and the radiation pipeline must read the SAME halo files "
                "(check MetallicityEvolution.halo_subdir / halo_filename_fmt against "
                "Output.sources_basename)."
            )
        return np.asarray(halos["f_Z_halo"], dtype=float)

    def _read_f_Z_cell(self, i: int):
        """Flat f_Z_cell array (length N_cell^3) from snapshot i's gridded file,
        or None if the file is absent (e.g. first slice / not yet produced)."""
        path = self.gridded_metallicity_file(i)
        if not path.exists():
            return None
        return np.load(path, allow_pickle=True)["f_Z_cell"]

    def prepare_cellwise_slice(self, srcpos, i: int) -> bool:
        """Per-bin cell-by-cell setup: give each source its metallicity and assign
        it to the BPASS metallicity bins by a log-Z FLUX SPLIT across its two
        bracketing bins (weights 1-w and w). Caches, for each occupied bin, the
        contributing source indices and their split weights.

        The source metallicity comes from self.src_fz when ionizing_flux was given
        the snapshot index (PREFERRED: the per-halo f_Z of snapshot i binned on
        the same mesh as the stellar mass, so source and metallicity are the same
        cell by construction; those per-halo values were themselves inherited
        from snapshot i-1's gridded field, so no star sees its own metals).
        Otherwise it falls back to indexing the f_Z_cell
        grid with the source's mesh index, which is only correct if the source
        mesh and the metallicity mesh share a cell width -- they do not, since
        bin_sources is called with meshsize=N+1 while f_Z_cell is on N^3. That
        fallback therefore mis-registers most sources and warns.

        Each occupied bin becomes one raytracing group in evolve3D_cellwise, with
        that bin's fixed table (spectral shape) and its bin-center q_ion
        (amplitude). Returns True if active; False -> volume-average fallback
        (first slice / missing metallicity data / no sources).
        """
        # Any earlier slice's mapping is invalid from here on: drop it first so a
        # fallback return can never leave stale bins behind for evolve3D_cellwise.
        self._cw_bins, self._cw_nsrc = {}, 0

        if not self.cellwise or self.metallicity_params is None:
            return False

        # ionizing_flux always returns (n_src, 3) from bin_sources; assert rather
        # than infer the orientation, which is ambiguous when n_src == 3.
        srcpos = np.asarray(srcpos)
        if srcpos.ndim != 2 or srcpos.shape[1] != 3:
            raise ValueError(
                "prepare_cellwise_slice expects srcpos with shape (n_src, 3), got "
                f"{srcpos.shape}"
            )
        if srcpos.shape[0] == 0:
            return False

        Z_min = float(self.metallicity_params.Z_min)

        if self.src_fz is not None:
            # Preferred path: metallicity binned alongside the stellar mass.
            if self.src_fz.shape[0] != srcpos.shape[0]:
                raise RuntimeError(
                    f"src_fz has {self.src_fz.shape[0]} entries but srcpos has "
                    f"{srcpos.shape[0]}: call prepare_cellwise_slice on the srcpos "
                    "returned by the ionizing_flux call that built src_fz."
                )
            fz_src = np.maximum(self.src_fz, Z_min)
        else:
            # Legacy fallback: index the gridded field with the source's mesh
            # index. Kept so a driver that does not pass i still runs.
            f_Z_cell = self._read_f_Z_cell(i - 1)
            if f_Z_cell is None:
                logger.info(
                    "Cellwise: no per-halo or gridded metallicity for snapshot %d "
                    "-> volume-average fallback.", i - 1,
                )
                return False

            N = self.N
            assert f_Z_cell.size == N**3, (
                f"Cellwise requires the metallicity grid ({f_Z_cell.size} cells) to "
                f"match the RT mesh N^3={N**3}. Regrid, or set N_cell == meshsize."
            )

            _, fz_src = classify_source_cells(srcpos, f_Z_cell, N, Z_min)
            logger.warning(
                "Cellwise slice %d is using the LEGACY cell-grid metallicity "
                "lookup: source mesh indices (bin_sources meshsize=N+1) are being "
                "used to index an N^3 grid, so most sources read a neighbouring "
                "cell and land at the Z_min floor. Pass i= to ionizing_flux to use "
                "the registered per-halo metallicity instead.", i,
            )

        # Bracketing BPASS bins + log-Z split weight per source.
        zc = self.ztables.Z_bin_centers
        lo, hi, w = bin_weights(fz_src, zc)

        n = srcpos.shape[0]
        src_idx = np.arange(n)
        # Each source contributes to its low bin (weight 1-w) and high bin (w).
        all_idx = np.concatenate([src_idx, src_idx])
        all_bin = np.concatenate([lo, hi])
        all_wt = np.concatenate([1.0 - w, w])
        keep = all_wt > 0.0
        all_idx, all_bin, all_wt = all_idx[keep], all_bin[keep], all_wt[keep]

        self._cw_bins = {
            int(b): (all_idx[all_bin == b], all_wt[all_bin == b])
            for b in np.unique(all_bin)
        }
        self._cw_nsrc = n

        n_enriched = int((fz_src > Z_min * (1.0 + 1e-9)).sum())
        logger.info(
            "Cellwise per-bin slice %d: %d sources across %d occupied BPASS bin(s) "
            "(%d enriched, %d at floor Z=%.1e).",
            i, n, len(self._cw_bins), n_enriched, n - n_enriched, Z_min,
        )
        return True

    def evolve3D_cellwise(self, dt: float, src_pos, age_lo_seconds: float, age_hi_seconds: float) -> None:
        """Per-bin evolve: each occupied BPASS metallicity bin is a raytracing
        group with its own fixed table (shape) and its bin-center q_ion
        (amplitude), interval-averaged over the sub-step [age_lo, age_hi]. Sources
        are flux-split across bins per prepare_cellwise_slice. Up to n_bins passes
        per convergence iteration (~2x the raytracing work: each source touches 2
        bins). Shared chemistry over the summed rate (evolve3D_multigroup).
        """
        if self.src_mstar is None:
            raise RuntimeError("evolve3D_cellwise needs src_mstar from ionizing_flux")
        if not self._cw_bins:
            raise RuntimeError(
                "evolve3D_cellwise called with no active cellwise slice: "
                "prepare_cellwise_slice must return True for this slice first "
                "(it returns False on the first slice / a missing gridded file / "
                "no sources, in which case the driver must take the "
                "volume-average path instead)."
            )
        if self.src_mstar.shape[0] != self._cw_nsrc:
            raise RuntimeError(
                f"cellwise source count changed ({self._cw_nsrc} at "
                f"prepare_cellwise_slice, {self.src_mstar.shape[0]} now): the "
                "cached bin mapping indexes a different source list. Call "
                "prepare_cellwise_slice after ionizing_flux for every slice."
            )

        # ionizing_flux returns (n_src, 3); assert instead of inferring, which is
        # ambiguous when n_src == 3.
        src_pos = np.asarray(src_pos)
        if src_pos.ndim != 2 or src_pos.shape[1] != 3:
            raise ValueError(
                "evolve3D_cellwise expects src_pos with shape (n_src, 3), got "
                f"{src_pos.shape}"
            )
        src_pos = src_pos.T  # -> (3, n_src), as evolve3D_multigroup expects

        groups = []
        for b, (idx, wt) in self._cw_bins.items():
            Z_b = float(self.ztables.Z_bin_centers[b])
            q_b = self._qion_rate_interval(Z_b, age_lo_seconds, age_hi_seconds)
            normflux_b = self.src_mstar[idx] * wt * q_b / 1e48
            thin_b = np.ascontiguousarray(self.ztables.photo_thin[b])
            thick_b = np.ascontiguousarray(self.ztables.photo_thick[b])
            groups.append((src_pos[:, idx], normflux_b, thin_b, thick_b))

        NumSrc = self.src_mstar.shape[0]
        use_mpi = NumSrc >= self.nprocs and self.mpi
        self.xh, self.phi_ion = evolve3D_multigroup(
            dt=dt,
            dr=self.dr,
            groups=groups,
            src_batch_size=self.raytracing_params.source_batch_size,
            use_gpu=self.gpu,
            max_subbox=self.max_subbox,
            subboxsize=self.subboxsize,
            loss_fraction=self.loss_fraction,
            use_mpi=use_mpi,
            rank=self.rank if use_mpi else 0,
            nprocs=self.nprocs if use_mpi else 1,
            temp=self.temp,
            ndens=self.ndens,
            xh=self.xh,
            clump=self.clumping_factor,
            minlogtau=self.minlogtau,
            dlogtau=self.dlogtau,
            R_max_LLS=self.R_max_LLS,
            convergence_fraction=self.convergence_fraction,
            sig=self.sig,
            bh00=self.bh00,
            albpow=self.albpow,
            colh0=self.colh0,
            temph0=self.temph0,
            abu_c=self.abu_c,
        )

        # evolve3D_multigroup leaves the LAST group's table on the device. Restore
        # the single active table so any later code that assumes the device
        # matches self.photo_thin/thick_table (base evolve3D, or a volume-average
        # fallback slice) does not silently raytrace with a stray Z bin's
        # spectrum.
        if self.gpu:
            photo_table_to_device(self.photo_thin_table, self.photo_thick_table)

# ---------------------------------------------------------------------------
# Lightweight metallicity-evolution helper classes (integrated from draft)
# ---------------------------------------------------------------------------


def classify_source_cells(srcpos_ijk, f_Z_cell, N: int, Z_min: float):
    """Classify sources as 'old' (previously enriched) by their cell metallicity.

    Parameters
    ----------
    srcpos_ijk : (n_src, 3) array of 0-indexed RT-grid cell coordinates.
    f_Z_cell : flat (N**3,) per-cell metallicity grid (C-order: idx = i*N^2+j*N+k).
    N : mesh size (metallicity grid assumed to match the RT mesh).
    Z_min : base metallicity floor; cells above it are 'old' (enriched).

    Returns
    -------
    mask_old : (n_src,) bool, True where the source's cell f_Z_cell > Z_min.
    fz_src : (n_src,) the f_Z_cell value at each source's cell.
    """
    srcpos_ijk = np.asarray(srcpos_ijk)
    ii = np.clip(srcpos_ijk[:, 0].astype(int), 0, N - 1)
    jj = np.clip(srcpos_ijk[:, 1].astype(int), 0, N - 1)
    kk = np.clip(srcpos_ijk[:, 2].astype(int), 0, N - 1)
    fz_src = np.asarray(f_Z_cell)[ii * N * N + jj * N + kk]
    return fz_src > Z_min * (1.0 + 1e-9), fz_src


def bin_weights(Z_src, Z_bin_centers):
    """Vectorized log-Z flux-split weights for the per-bin cellwise regime.

    For each metallicity in ``Z_src`` return the two bracketing BPASS bin indices
    (lo, hi = lo+1) and the log-Z weight ``w`` in [0, 1]: the source contributes a
    fraction ``1-w`` of its stellar mass to bin ``lo`` and ``w`` to bin ``hi``.
    Z is clamped to the bin range (w=0 at/below the lowest bin, w=1 at/above the
    highest), so no source is ever extrapolated beyond the tabulated bins.

    Parameters
    ----------
    Z_src : (n_src,) source metallicities (e.g. f_Z_cell).
    Z_bin_centers : (n_bins,) sorted BPASS bin metallicities.

    Returns
    -------
    lo, hi : (n_src,) int arrays of bracketing bin indices (hi = lo + 1).
    w : (n_src,) float split weight toward the high bin.
    """
    zc = np.asarray(Z_bin_centers, dtype=float)
    Z = np.clip(np.asarray(Z_src, dtype=float), zc[0], zc[-1])
    hi = np.clip(np.searchsorted(zc, Z), 1, zc.size - 1)
    lo = hi - 1
    w = (np.log10(Z) - np.log10(zc[lo])) / (np.log10(zc[hi]) - np.log10(zc[lo]))
    return lo, hi, np.clip(w, 0.0, 1.0)


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
    """Standalone single-power-law stellar-to-halo relation.

    WARNING: this does NOT reproduce any of the Sources.fstar_kind models used
    by the ionizing-source pipeline. In particular it has the OPPOSITE mass
    slope to the 'dpl' relation (alpha_star = -0.3 here vs +0.3 effective for
    g1 = g2 = -0.3) and a ~10x larger amplitude at 1e10 Msun. Using it while the
    radiation side runs 'dpl' means the same halo produces metals and photons
    from two different stellar masses. Prefer make_stellar_mass_model(), which
    mirrors the configured Sources block; this class is kept only for the legacy
    calibration and as a fallback for fstar_kind values with no deterministic
    mean (e.g. the stochastic 'lognorm' / 'Muv' variants).

    Returns M_star in Msun/h (same units as the input halo mass), matching
    MetallicityEvolution.baryonic_mass_per_cell so that f_Z = M_Z / M_b is
    unit-consistent.
    """

    f_star_zero: float = 0.02
    alpha_star: float = -0.3
    pivot_mass: float = 1e10

    def __call__(self, halo_mass_msunh: np.ndarray, h: float) -> np.ndarray:
        halo_mass_msun = halo_mass_msunh * h
        f_star = self.f_star_zero * (halo_mass_msun / self.pivot_mass) ** self.alpha_star
        return f_star * halo_mass_msunh


@dataclass
class StellarMassModelDPL:
    """Double-power-law stellar-to-halo relation, mirroring the 'dpl' branch of
    StellarToHaloRelation.stellar_to_halo_fraction (2011.12308, 2201.02210,
    2302.06626) so that the enrichment pipeline and C2Ray_Metals.ionizing_flux
    assign the SAME stellar mass to a given halo.

    The functional form is kept identical to source_model.py on purpose -- if
    that relation changes, change it here too.

    Units: takes M_halo in Msun/h (as stored in the PKDGrav halo txt files),
    converts to Msun for the f_star evaluation (source_model.py is fed the
    h-corrected masses from read_haloes), and returns M_star in Msun/h so that
    MetallicityEvolution.baryonic_mass_per_cell -- which is also in Msun/h --
    divides out consistently in f_Z_cell.
    """

    f0: float
    Mp: float
    g1: float
    g2: float
    Mt: float
    g3: float
    g4: float
    Omega_b: float
    Omega_m: float

    def __call__(self, halo_mass_msunh: np.ndarray, h: float) -> np.ndarray:
        M_msunh = np.asarray(halo_mass_msunh, dtype=float)
        M = M_msunh / h  # Msun, as passed to StellarToHaloRelation.get

        dpl = (
            2.0
            * self.Omega_b
            / self.Omega_m
            * self.f0
            / ((M / self.Mp) ** self.g1 + (M / self.Mp) ** self.g2)
        )

        # Suppression at the small-mass end
        S_M = (1 + (self.Mt / M) ** self.g3) ** self.g4

        return dpl * S_M * M_msunh


def make_stellar_mass_model(sources_params, Omega_b: float, Omega_m: float):
    """Pick the stellar-mass model for the enrichment pipeline from the Sources
    block, so it agrees with the relation the radiation pipeline actually uses.

    'dpl' (and the 'spice' variants, which share the same deterministic mean)
    map onto StellarMassModelDPL. Anything else has no deterministic mean that
    can be reused halo-by-halo here, so we fall back to StellarMassModelLite and
    warn -- the metals and the photons will then come from different stellar
    masses.
    """
    kind = getattr(sources_params, "fstar_kind", None)

    if kind == "dpl" or (isinstance(kind, str) and "spice" in kind):
        return StellarMassModelDPL(
            f0=sources_params.f0,
            Mp=sources_params.Mp,
            g1=sources_params.g1,
            g2=sources_params.g2,
            Mt=sources_params.Mt,
            g3=sources_params.g3,
            g4=sources_params.g4,
            Omega_b=Omega_b,
            Omega_m=Omega_m,
        )

    logger.warning(
        "Sources.fstar_kind='%s' has no deterministic stellar-to-halo mean to "
        "reuse for enrichment; falling back to StellarMassModelLite. The metals "
        "and the ionizing photons will be computed from DIFFERENT stellar masses.",
        kind,
    )
    return StellarMassModelLite()


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

        # lifetime metal yield y(Z); built lazily, only for runs that need it
        self._y_total: np.ndarray | None = None

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

    def total_metal_yield(self) -> np.ndarray:
        """y(Z): metal mass (winds + SN) produced per unit INITIAL stellar mass,
        integrated over the FULL BPASS age grid.

        This is deliberately NOT ``F_table[:, -1]``. That stops at ``t_max_myr``
        (the 10 Myr burst window) and so captures only the wind phase -- core
        collapse SNe from stars below ~20 Msun have not exploded by then, and
        they dominate the metal budget. y(Z) is the lifetime-integrated quantity
        needed for an equilibrium ISM enrichment estimate.

        Integration is a rectangle sum over BPASS's own log-spaced bin widths
        (dt_i = 10^(log_age_i + d/2) - 10^(log_age_i - d/2), d the log spacing),
        not a trapezoid between bin centres: over five decades of log-spaced bins
        the trapezoid is badly biased and drops the first and last half-bins.
        """
        log_age = np.loadtxt(self.bpass_root / "ages.txt")
        if log_age.size < 2:
            raise ValueError(f"ages.txt needs >= 2 entries, got {log_age.size}")
        spacing = np.diff(log_age)
        if not np.allclose(spacing, spacing[0], rtol=1e-6):
            raise ValueError(
                "total_metal_yield assumes a uniformly log-spaced BPASS age grid; "
                f"ages.txt spacing ranges {spacing.min():.4f}..{spacing.max():.4f} dex."
            )
        half = 0.5 * float(spacing[0])
        dt_yr = 10.0 ** (log_age + half) - 10.0 ** (log_age - half)

        y = np.empty(len(self.Z_list), dtype=float)
        for i, Z in enumerate(self.Z_list):
            yields = self._load_yields(Z)
            rate = (yields["Z_sw"] + yields["Z_sn"]).values  # Msun/yr per M_BPASS
            n = min(rate.size, dt_yr.size)
            y[i] = float(np.sum(rate[:n] * dt_yr[:n]) / self.M_BPASS)
        return y

    @property
    def y_total(self) -> np.ndarray:
        """Cached lifetime metal yield per BPASS metallicity bin (see
        total_metal_yield). Built on first use so runs that do not enable ISM
        self-enrichment never pay for it."""
        if self._y_total is None:
            self._y_total = self.total_metal_yield()
        return self._y_total

    def y_of_Z(self, metallicity):
        """Lifetime metal yield y(Z), log-Z interpolated between the bracketing
        BPASS bins (same interpolation as metal_produced / mass_returned)."""
        lo, hi, w = self._z_weights(metallicity)
        y = self.y_total
        return (1.0 - w) * y[lo] + w * y[hi]

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
        ism_self_enrichment: bool = False,
        metal_retention: float = 0.5,
        burst_fraction=None,
    ) -> None:
        self.cosmology = cosmology
        self.paths = paths
        self.bpass = bpass if bpass is not None else BPASSYieldTable(Path("."))
        self.stellar_mass_model = stellar_mass_model
        self.N_cell = N_cell
        self.Z_min = Z_min
        self.ism_self_enrichment = bool(ism_self_enrichment)
        self.metal_retention = float(metal_retention)
        # Optional callable snapshot_index -> fraction of the halo's cumulative
        # stellar mass that formed during this snapshot. None => 1.0 (previous
        # behaviour: every star treated as newly formed at every snapshot).
        self.burst_fraction = burst_fraction
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

        # Environmental term: the metallicity of the gas the halo accretes, i.e.
        # its cell's metallicity from the previous snapshot. This is the whole of
        # f_Z_halo in the default configuration.
        if i == i_start:
            f_Z_env = np.full(len(halos), self.Z_min, dtype=float)
        else:
            prev = np.load(self.paths.gridded_file(i - 1), allow_pickle=True)
            f_Z_env = np.asarray(prev["f_Z_cell"][halos["cell_id"]], dtype=float)

        if self.ism_self_enrichment:
            # Self-enrichment: metals this halo produced over its lifetime and
            # retained, diluted in its own baryon budget f_b * M_halo.
            #
            # M_star here is the halo's CUMULATIVE stellar mass, which is the
            # correct quantity for accumulated chemical enrichment (unlike the
            # instantaneous ionizing output, which needs a star formation rate).
            # y(Z) is lifetime-integrated, NOT the 10 Myr burst-window yield --
            # a galaxy's ISM carries metals from every generation it has formed,
            # which is a different clock from the 10 Myr radiation window.
            #
            # accumulate_metals is deliberately left untouched: the two terms are
            # integrals over different time windows, so partitioning one against
            # the other is not well posed, and the overlap (this halo's own
            # contribution to its ~Mpc cell) is sub-percent.
            f_b = self.cosmology.Omega_b / self.cosmology.Omega_m
            gas = f_b * np.asarray(halos["Mhalo"], dtype=float)
            ratio = np.divide(
                np.asarray(halos["M_star"], dtype=float),
                gas,
                out=np.zeros_like(gas),
                where=gas > 0.0,
            )
            y = self.bpass.y_of_Z(np.maximum(f_Z_env, self.Z_min))
            halos["f_Z_halo"] = f_Z_env + y * self.metal_retention * ratio
        else:
            halos["f_Z_halo"] = f_Z_env

        return halos

    def evolve(self, halos: np.ndarray, i: int, t1_idx: int, t2_idx: int, save: bool = True) -> np.ndarray:
        out = halos.copy()

        # Fraction of the cumulative M_star that is actually NEW this snapshot.
        # BPASS yields are per unit mass of a coeval population, so only the
        # newly formed stars should produce metals in this window; without this
        # the same stars are re-created and re-enrich at every snapshot.
        frac = 1.0
        if self.burst_fraction is not None:
            frac = float(self.burst_fraction(i))
            logger.info(
                "Snapshot %d: SFR-normalised yield, burst fraction = %.4f "
                "(metals from the stars formed in this snapshot only)", i, frac,
            )

        # Metals deposited into the gas: metals-only (Z winds + SN) channel.
        delta_Z = frac * self.bpass.metal_produced(
            out["f_Z_halo"], out["M_star"], t1_idx=t1_idx, t2_idx=t2_idx
        )
        # Mass leaving the stars: FULL H + He + Z (winds + SN) ejecta. The
        # returned H/He rejoins gas whose mass is already set by the
        # overdensity grid, so only delta_Z enters the metal budget. Scaled by
        # the same fraction: this whole step describes the young population.
        delta_ej = frac * self.bpass.mass_returned(
            out["f_Z_halo"], out["M_star"], t1_idx=t1_idx, t2_idx=t2_idx
        )
        out["M_delta_Z"] = delta_Z
        out["M_star"] -= delta_ej

        if save:
            np.save(self.paths.metallicity_file(i), out)
        return out

    def baryonic_mass_per_cell(self, overdensity: np.ndarray) -> np.ndarray:
        """Baryonic mass per cell, in Msun/h to match the stellar masses.

        NOTE the Omega_b / Omega_m factor is correct and deliberate: ``overdensity``
        here is the RAW PKDGrav3 .npy field, which stores rho_m / rho_crit,0
        (box mean = Omega_m), not 1 + delta. So

            rho_c * (Omega_b/Omega_m) * (Omega_m * (1+delta)) * V
                = rho_c * Omega_b * (1+delta) * V

        which is the baryon mass. Do NOT "fix" this by dropping the /Omega_m
        without also changing what is passed in -- C2Ray_Metals.read_density
        divides the same field by Omega_m before use, precisely because it feeds
        an expression that supplies Omega_b on its own.
        """
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
