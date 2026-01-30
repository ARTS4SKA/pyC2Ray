from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from typing import Any, Optional, Union

import yaml

import pyc2ray.constants as c

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

# Configure YML to read scientific notation as floats rather than strings
YML_REGEX = re.compile(
    """^(?:
[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
|\\.[0-9_]+(?:[eE][-+][0-9]+)?
|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
|[-+]?\\.(?:inf|Inf|INF)
|\\.(?:nan|NaN|NAN))$""",
    re.X,
)
SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", YML_REGEX, list("-+0123456789.")
)

PathType = Union[str, os.PathLike]
OptFloat = Optional[float]
OptStr = Optional[str]


@dataclass
class YmlParameters:
    @classmethod
    def load_yaml(cls, file: PathType) -> dict[str, Any]:
        """Read in YAML parameter file"""
        with open(file, "r") as f:
            return yaml.load(f, SafeLoader)

    @classmethod
    def from_dict(cls, yml: dict[str, Any]) -> YmlParameters:
        keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in yml.items() if k in keys})

    @classmethod
    def from_file(cls, file: PathType, block: Optional[str] = None) -> YmlParameters:
        """Read in YAML parameter file"""
        ld = cls.load_yaml(file)
        if block is not None:
            ld = ld.get(block, {})
        return cls.from_dict(ld)


@dataclass
class OutputParameters(YmlParameters):
    """Output setup"""

    # Directory where results and log files are stored
    results_basename: str
    # Directory where input files are stored
    inputs_basename: OptStr = None
    # Basename of the sources file
    sources_basename: OptStr = None
    # Basename of the density file
    density_basename: OptStr = None
    # Name of the log file to write
    logfile: str = "pyC2Ray.log"


@dataclass
class GridParameters(YmlParameters):
    """Parameters to set up the simulation volume"""

    # Box size in comoving Mpc
    boxsize: float
    # Side of the mesh grid
    meshsize: int
    # Use GPU acceleration
    gpu: bool
    # Use MPI parallelization
    mpi: bool
    # Resume a simulation
    resume: bool


@dataclass
class RaytracingParameters(YmlParameters):
    """ASORA Raytracing parameters"""

    # Photon loss fraction for the subbox algorithm
    loss_fraction: float
    # Size increase of subboxes around sources
    subboxsize: int
    # Maximum subbox size for the subbox algorithm
    max_subbox: int
    # Number of sources to be processed in parallel
    source_batch_size: int
    # Which fraction of the cells can be left unconverged
    convergence_fraction: float


@dataclass
class MaterialParameters(YmlParameters):
    """Properties of physical quantities in the simulation volume"""

    # Initial Temperature of the grid
    temp0: float
    # Initial Ionized fraction of the grid
    xh0: float
    # Constant average density, comoving value
    avg_dens: float


@dataclass
class CGSParameters(YmlParameters):
    """Miscellaneous physical constants"""

    # Hydrogen recombination parameter (power law index)
    albpow: float
    # Hydrogen recombination parameter (value at 10^4 K)
    bh00: float
    # Helium0 recombination parameter (power law index)
    alcpow: float
    # Hydrogen ionization energy (in eV)
    eth0: float = 13.598
    # Helium I ionization energy (in eV)
    ethe0: float = 24.587
    # Helium II ionization energy (in eV)
    ethe1: float = 54.416
    # Hydrogen collisional ionization parameter 1
    xih0: float = 1.0
    # Hydrogen collisional ionization parameter 2
    fh0: float = 0.83
    # Colfh0 factor
    colh0_fact: float = 1.3e-8

    def __post_init__(self):
        self.colh0 = self.colh0_fact * self.fh0 * self.xih0 / self.eth0**2
        self.temph0 = self.eth0 * c.ev2k


@dataclass
class CosmologyParameters(YmlParameters):
    """Cosmological Parameters"""

    # Global flag to use cosmology
    cosmological: bool
    # Reduced Hubble constant
    h: float
    # Omega matter t=0
    Omega0: float
    # Omega baryon t=0
    Omega_B: float
    # Initial redshift of the simulation
    zred_0: float
    # Temperature of CMB in Kelvin
    cmbtemp: float = 2.726


@dataclass
class AbundancesParameters(YmlParameters):
    """Element abundances"""

    # Hydrogen Abundance
    abu_h: float = 0.926
    # Helium Abundance
    abu_he: float = 0.074
    # Carbon Abundance
    abu_c: float = 7.1e-7

    def __post_init__(self):
        self.mean_molecular = self.abu_h + 4.0 * self.abu_he


@dataclass
class PhotoParameters(YmlParameters):
    """Parameters governing photoionization"""

    # HI cross section at its ionizing frequency (weighted by freq_factor)
    sigma_HI_at_ion_freq: float
    # Minimum optical depth for tables
    minlogtau: float
    # Maximum optical depth for tables
    maxlogtau: float
    # Number of table points
    NumTau: int
    # Whether or not to use grey opacity (i.e. cross-section is frequency-independent)
    grey: bool
    # Type of source to use
    SourceType: str
    # Whether to compute heating rates arrays (NOT USED BY CHEMISTRY SO FAR)
    compute_heating_rates: bool
    # Name of the SED table file
    sed_table: str = ""


@dataclass
class SinksParameters(YmlParameters):
    """Parameters for sinks"""

    # Clumping model, values are "constant", "redshift", "density" or "stochastic"
    clumping_model: str
    # Mean-free-path model "constant", "Choudhury09"
    mfp_model: str
    # Clumping factor for the constant model
    clumping: OptFloat = None
    # Maximum comoving distance for photons from source
    R_max_cMpc: OptFloat = None
    # Free parameter for the Choudhury09 mean-free-path model in cMpc units
    A_mfp: OptFloat = None
    # Spectral index of the Choudhury09 mean-free-path model redshift evolution
    eta_mfp: OptFloat = None
    # TODO: add a description
    z1_mfp: OptFloat = None
    # TODO: add a description
    eta1_mfp: OptFloat = None

    def __post_init__(self) -> None:
        if self.clumping_model not in ("constant", "redshift", "density", "stochastic"):
            raise ValueError(
                f"Clumping model {self.clumping_model} not implemented. "
                "Choose from 'constant', 'redshift', 'density' or 'stochastic'."
            )
        if self.mfp_model not in ("constant", "Choudhury09"):
            raise ValueError(
                f"Mean-free-path model {self.mfp_model} not implemented. "
                "Choose from 'constant' or 'Choudhury09'."
            )


@dataclass
class BlackBodyParameters(YmlParameters):
    """Parameters for Black Body source type"""

    # Effective temperature of Black Body source
    Teff: float
    # Power-law index for the frequency dependence of the photoionization cross section
    cross_section_pl_index: float


@dataclass
class SourcesParameters(YmlParameters):
    """Parameters for sources"""

    # stellar-to-halo mass relation:
    #  'fgamma' for classical mass independent model
    #  'dpl' for double power law (Schneider, Giri, Mirocha 2021)
    #  'lognorm' for a stochastic with lognorm distribution and std ~Mhalo^(-1/3).
    #  'Muv' for a scatter in the absolute magnitude with std_UV~a_s*log10(Mhalo)+b_s (see Gelli+ 2024)
    fstar_kind: str = "fgamma"
    # efficiency High-Mass Atomically Cooling Halo (HMACH) - used only for fstar_kind: 'fgamma'
    fgamma_hm: OptFloat = None
    # efficiency Low-Mass Atomically Cooling Halo (LMACH)
    fgamma_lm: OptFloat = None
    # Double power law parameter - these are used only when fstar_kind: 'dpl'
    Nion: OptFloat = None
    f0: OptFloat = None
    Mt: OptFloat = None
    Mp: OptFloat = None
    g1: OptFloat = None
    g2: OptFloat = None
    g3: OptFloat = None
    g4: OptFloat = None
    # Free parameters for the Muv-scatter in the fstar model, relevant for fstar_kind == 'Muv'
    a_s: OptFloat = None
    b_s: OptFloat = None
    # Photons escaping fraction:
    #  'constant' for mass independent model,
    #  'power' for a power law mass dependent factor
    #  'Gelli2024' for a UV dependent model (Gelli+ 2024)
    fesc_model: str = "constant"
    f0_esc: OptFloat = None
    Mp_esc: OptFloat = None
    al_esc: OptFloat = None
    # define the accretion model: 'constant' or 'EXP'
    accretion_model: str = "constant"
    # accretion rate parameter (see Schneider+ 2021)
    alpha_h: OptFloat = None
    # bursty star-formation model, requires accretion model 'EXP':
    #  'no'
    #  'instant'
    #  'integrate', it
    bursty_sfr: str = "no"
    # index power-low of the bursty star-formation model mass relation
    beta1: OptFloat = None
    # index power-low of the bursty star-formation model time relation
    beta2: OptFloat = None
    # bursty star-formation time-scale at z=0
    tB0: OptFloat = None
    # fraction of the quiescent time-scale
    tQ_frac: OptFloat = None
    # reference redshift for the bursty star-formation model
    z0: OptFloat = None
    # Randomize the time-scale of the bursty star-formation model (std for N~(t_start, t_rnd))
    t_rnd: OptFloat = None
    # TODO: add description
    ts: OptFloat = None

    def __post_init__(self) -> None:
        if self.fstar_kind not in ("fgamma", "dpl", "lognorm", "Muv"):
            raise ValueError(
                f"fstar_kind {self.fstar_kind} not implemented. "
                "Choose from 'fgamma', 'dpl', 'lognorm' or 'Muv'."
            )
        if self.fesc_model not in ("constant", "power", "Gelli2024"):
            raise ValueError(
                f"fesc_model {self.fesc_model} not implemented. "
                "Choose from 'constant', 'power' or 'Gelli2024'."
            )
        if self.accretion_model not in ("constant", "EXP"):
            raise ValueError(
                f"accretion_model {self.accretion_model} not implemented. "
                "Choose from 'constant' or 'EXP'."
            )
        if self.bursty_sfr not in ("no", "instant", "integrate"):
            raise ValueError(
                f"bursty_sfr {self.bursty_sfr} not implemented. "
                "Choose from 'no', 'instant' or 'integrate'."
            )
