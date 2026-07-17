from __future__ import annotations

import logging
import os
import re
from dataclasses import KW_ONLY, MISSING, InitVar, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

import pyc2ray.constants as c


class _Loader(yaml.SafeLoader):
    """A `SafeLoader` with the scientific-notation float fix applied.

    Subclassing rather than patching `yaml.SafeLoader` directly keeps the fix
    scoped to this module's own YAML loading, instead of mutating a shared
    class from a third-party library for every other user of `yaml` in the
    same process.
    """

    YML_REGEX = re.compile(
        """^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    )


def yaml_format(key: str, value: Any) -> str:
    """Wrapper around `yaml.safe_dump` that uses `_Loader` to read floats in scientific notation."""
    if isinstance(value, Path):
        value = str(value)
    return yaml.safe_dump(
        {key: value}, default_flow_style=False, sort_keys=False
    ).strip()


# Configure YML to read scientific notation as floats rather than strings
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _Loader.YML_REGEX, list("-+0123456789.")
)

PathType = str | os.PathLike
ParametersType = TypeVar("ParametersType", bound="YmlParameters")

logger = logging.getLogger(__name__)


def _field(doc: str, default: Any = MISSING) -> Any:
    """Shorthand for a dataclass field carrying a human-readable description.
    If default is not provided, the field is required and must be specified in the input YAML file.

    The description is used both as a source-level docstring substitute and,
    via `YmlParameters.to_yaml_block`, as the comment written above the
    corresponding key when generating a parameters YAML file.
    """
    # NOTE: replace metadata with doc from python 3.14
    return field(default=default, metadata={"doc": doc})


# Sentinel object to detect whether a YmlParameters subclass was constructed
# directly (not allowed) or via a factory method (allowed).
_factory = object()


@dataclass
class YmlParameters(Generic[ParametersType]):
    SECTION: ClassVar[str]
    _: KW_ONLY
    _build: InitVar[object] = field(default=None, compare=False, repr=False)

    def _validate_literals(self) -> None:
        """Raise ValueError if any Literal-typed field's value isn't one of its choices."""
        hints = get_type_hints(type(self))
        for f in fields(self):
            type_hint = hints[f.name]
            origin = get_origin(type_hint)
            if origin is not Literal:
                continue
            args = get_args(type_hint)
            value = getattr(self, f.name)
            if value not in args:
                raise ValueError(
                    f"Parameter {f.name} = {value!r} is not valid for {type(self).__name__}. "
                    f"Choose from {args!r}."
                )

    def __post_init__(self, _build: object) -> None:
        if _build is not _factory:
            raise TypeError(
                f"{type(self).__name__} must be created using factory methods, "
                "from_dict or from_file, not direct instantiation"
            )
        self._validate_literals()

    @classmethod
    def from_dict(
        cls: type[ParametersType], yml: dict[str, Any], warn: bool = False
    ) -> ParametersType:
        """Instantiate from a dict, optionally warning about missing keys.

        Parameters
        ----------
        yml : Dictionary containing the parameters for this section.
        warn : Whether to warn about using default values in the input dictionary.
        """

        keys = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in yml.items() if k in keys}, _build=_factory)
        if warn:
            missing_keys = keys - yml.keys()
            for key in missing_keys:
                logger.warning(
                    f"Using default value for parameter '{cls.SECTION}/{key}' = {getattr(obj, key)}"
                )
        return obj

    @classmethod
    def load_yaml(cls: type[ParametersType], file: PathType) -> dict[str, Any]:
        """Read in YAML parameter file"""
        with open(file, "r") as f:
            return yaml.load(f, _Loader)

    @classmethod
    def from_file(cls: type[ParametersType], file: PathType) -> ParametersType:
        """Read in YAML parameter file, optionally extracting a nested block of parameters."""
        yml = cls.load_yaml(file)
        return cls.from_dict(yml.get(cls.SECTION, {}))

    def to_yaml_block(self) -> str:
        """Render this section's current values as a commented YAML block."""
        lines: list[str] = []

        # Add the parameter class docstring as a header comment
        if self.__doc__ is not None:
            desc = self.__doc__.splitlines()[0].strip()
            sep = "# " + "-" * len(desc)
            lines.extend((sep, f"# {desc}", sep))

        # Add the section name
        lines.append(f"{self.SECTION}:")
        for f in fields(self):
            # Call to_yaml_block for nested YmlParameters
            attr = getattr(self, f.name)
            if isinstance(attr, YmlParameters):
                lines.append(attr.to_yaml_block())
                continue

            # Add the field's value with docstring as a comment
            if doc := f.metadata.get("doc"):
                lines.append(f"  # {doc}")
            lines.append("  " + yaml_format(f.name, attr))
        return "\n".join(lines)


RegistryModel = TypeVar("RegistryModel", bound="_ModelRegistry")


class _ModelRegistry(YmlParameters, Generic[RegistryModel]):
    """Base for a group of mutually-exclusive parameter sets selected by a
    string discriminator (e.g. `fstar_model`, `mfp_model`).

    Each subclass of this class should define its registry dictionary and
    a `KEY` class variable that specifies the name of the parameter set in
    the input parameter dictionary.

    Each concrete parameter "model" subclass sets instead the `MODEL`  class variable
    and is automatically registered to the parameter category's register via
    `__init_subclass__`.
    """

    _registry: ClassVar[dict[str, type[RegistryModel]]]
    MODEL: ClassVar[str]
    KEY: ClassVar[str]

    def __init_subclass__(cls: type[RegistryModel], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "MODEL"):
            cls._registry[cls.MODEL] = cls

    @classmethod
    def from_model(
        cls, yml: dict[str, Any], default: str, warn: bool = False
    ) -> RegistryModel:
        """Instantiate whichever registered subclass matches `model`.

        Reads only the keys relevant to that subclass out of `yml`.

        Parameters
        ----------
        yml : Dictionary containing the parameters for this section.
        default : Default model name to use if `model` is not present in `yml`.
        warn : Whether to warn about missing keys in the selected model's parameters.
        """
        try:
            model = yml.get(cls.KEY, default)
            subclass = cls._registry[model]
        except KeyError:
            raise ValueError(
                f"{model!r} is not implemented for {cls.__name__}. "
                f"Choose from {sorted(cls._registry)!r}."
            )
        return subclass.from_dict(yml, warn=warn)

    def to_yaml_block(self) -> str:
        """Render this model's current values as a commented YAML block."""
        lines: list[str] = []

        # Print model selector key
        if doc := type(self).__base__.__doc__:
            lines.append(f"  # {doc}")
        lines.append("  " + yaml_format(self.KEY, self.MODEL))

        # Print model's parameters
        for f in fields(self):
            if doc := f.metadata.get("doc"):
                lines.append(f"  # {doc}")
            lines.append("  " + yaml_format(f.name, getattr(self, f.name)))
        return "\n".join(lines)


@dataclass()
class InputOutputParameters(YmlParameters):
    """Input output setup"""

    SECTION: ClassVar[str] = "InputOutput"

    # Output
    results_dir: Path = _field("Directory where results and log files are stored")
    log_name: str = _field("Name of the log file in results_dir", "pyC2Ray.log")

    # Input, required only by some C2Ray classes
    # TODO: make inputs_dir deprecated
    inputs_dir: Path = _field("Directory where input files are stored", Path(""))
    sources_dir: Path = _field(
        "Absolute or relative path with respect to inputs_dir to the source files",
        Path(""),
    )
    density_dir: Path = _field(
        "Absolute or relative path with repsect to inputs_dir to the density files",
        Path(""),
    )

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.results_dir = Path(self.results_dir)

        self.inputs_dir = Path(self.inputs_dir)
        self.sources_dir = Path(self.sources_dir)
        self.density_dir = Path(self.density_dir)

        if not self.sources_dir.is_absolute():
            self.sources_dir = self.inputs_dir / self.sources_dir
        if not self.density_dir.is_absolute():
            self.density_dir = self.inputs_dir / self.density_dir

    @property
    def log_file(self) -> Path:
        """Return the full path to the log file."""
        return self.results_dir / self.log_name


@dataclass
class GridParameters(YmlParameters):
    """Parameters to set up the simulation volume"""

    SECTION: ClassVar[str] = "Grid"

    boxsize: float = _field("Box size in comoving Mpc")
    meshsize: int = _field("Side of the mesh grid")
    gpu: bool = _field("Use GPU acceleration")
    mpi: bool = _field("Use MPI parallelization")
    resume: bool = _field("Resume a simulation")


@dataclass
class RaytracingParameters(YmlParameters):
    """ASORA and C2Ray (legacy) raytracing parameters"""

    SECTION: ClassVar[str] = "Raytracing"

    source_batch_size: int = _field("Number of sources to be processed in parallel")
    loss_fraction: float = _field("Photon loss fraction for the subbox algorithm", 1e-2)
    subboxsize: int = _field("Size increase of subboxes around sources", 128)
    max_subbox: int = _field("Maximum subbox size for the subbox algorithm", 1000)


@dataclass
class DomainDecompositionParameters(YmlParameters):
    """Parameters for domain decomposition and source grouping"""

    SECTION: ClassVar[str] = "DomainDecomposition"

    enabled: bool = _field("Enable domain decomposition/source grouping", False)
    # TODO: evaluate if keeping the non incremental mode: in case of real cosmological simulations,
    # the non incremental mode can become extremely expensive.
    # - 'morton' fits the enclosing sphere of every trial group accurately
    # - 'morton_incremental' grows it incrementally instead, which is much faster to build
    #   but yields looser spheres and therefore larger local grids
    grouping_algorithm: Literal["morton", "morton_incremental"] = _field(
        "Source grouping/domain decomposition algorithm", "morton_incremental"
    )
    max_num_sources_per_group: int = _field(
        "Maximum number of sources in one source group", 1000
    )
    # TODO: morton_bits is a morton specific parameter, should be moved to a morton grouping parameter class
    morton_bits: int = _field("Number of bits per dimension for Morton ordering", 10)
    max_memory_cost_per_group: float = _field(
        "Maximum allowed memory cost per group in bytes", 50.0e9
    )

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        if self.max_num_sources_per_group <= 0:
            raise ValueError(
                f"max_num_sources_per_group must be a positive integer. "
                f"Provided value is {self.max_num_sources_per_group}."
            )
        from pyc2ray.domain.morton_grouping import MAX_MORTON_BITS

        if not 0 < self.morton_bits <= MAX_MORTON_BITS:
            raise ValueError(
                f"morton_bits must be an integer in [1, {MAX_MORTON_BITS}]. "
                f"Provided value is {self.morton_bits}."
            )
        if self.max_memory_cost_per_group <= 0:
            raise ValueError(
                f"max_memory_cost_per_group must be a positive number. "
                f"Provided value is {self.max_memory_cost_per_group}."
            )


# TODO: Propse to name it initial conditions?
@dataclass
class MaterialParameters(YmlParameters):
    """Properties of physical quantities in the simulation volume"""

    SECTION: ClassVar[str] = "Material"

    temp0: float = _field("Initial Temperature of the grid", 1e4)
    xHII: float = _field("Initial Ionized fraction of the grid", 1.0e-5)
    avg_dens: float = _field("Constant average density, comoving value", 1.87e-7)


@dataclass
class ChemistryParameters(YmlParameters):
    """Miscellaneous physical constants"""

    SECTION: ClassVar[str] = "Chemistry"

    ion_energy_HI: float = _field("Hydrogen I ionization energy (in eV)", 13.598)
    ion_energy_HeI: float = _field("Helium I ionization energy (in eV)", 24.587)
    ion_energy_HeII: float = _field("Helium II ionization energy (in eV)", 54.416)
    albpow: float = _field("Hydrogen recombination parameter (power law index)", -0.7)
    b0_HI: float = _field(
        "Hydrogen recombination parameter (value at 10^4 K)", 2.59e-13
    )
    alcpow: float = _field("Helium0 recombination parameter (power law index)", -0.67)
    xi_HI: float = _field("Hydrogen collisional ionization parameter 1", 1.0)
    f_HI: float = _field("Hydrogen collisional ionization parameter 2", 0.83)
    col_HI_fact: float = _field("Colf_HI factor", 1.3e-8)
    convergence_fraction: float = _field(
        "Which fraction of the cells can be left unconverged", 1.0e-4
    )

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.col_HI = self.col_HI_fact * self.f_HI * self.xi_HI / self.ion_energy_HI**2
        self.temp_HI = self.ion_energy_HI * c.ev2k


@dataclass
class CosmologyParameters(YmlParameters):
    """Cosmological Parameters"""

    SECTION: ClassVar[str] = "Cosmology"

    cosmological: bool = _field("Global flag to use cosmology", False)
    h: float = _field("Reduced Hubble constant", 0.7)
    Omega0: float = _field("Omega matter t=0", 0.27)
    Omega_B: float = _field("Omega baryon t=0", 0.043)
    cmbtemp: float = _field("Temperature of CMB in Kelvin", 2.726)
    # Deprecated
    zred_0: float = _field("Initial redshift of the simulation", 9.0)
    abu_h: float = _field("Hydrogen abundance", 0.926)
    abu_he: float = _field("Helium abundance", 0.074)
    abu_c: float = _field("Carbon abundance", 7.1e-7)

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.mean_molecular = self.abu_h + 4.0 * self.abu_he


class SourceTypeParameters(_ModelRegistry["SourceTypeParameters"]):
    """Source type model"""

    _registry: ClassVar[dict[str, type[SourceTypeParameters]]] = {}
    KEY: ClassVar[str] = "source_type_model"


@dataclass
class BlackBodySourceParameters(SourceTypeParameters):
    """Parameters for black body source type"""

    MODEL: ClassVar[str] = "blackbody"

    Teff: float = _field("Effective temperature of black body source", 5e4)
    cross_section_pl_index: float = _field(
        "Power-law index for the frequency dependence of the photoionization cross section",
        2.8,
    )
    sigma_HI_at_ion_freq: float = _field(
        "HI cross section at its ionizing frequency (weighted by freq_factor)", 6.30e-18
    )


@dataclass
class PowerLawSourceParameters(SourceTypeParameters):
    """Parameters for power-law source type"""

    MODEL: ClassVar[str] = "powerlaw"


@dataclass
class Zackrisson11SourceParameters(SourceTypeParameters):
    """Parameters for Zackrisson '11 source model"""

    MODEL: ClassVar[str] = "Zackrisson11"

    Teff: float = _field("Effective temperature of Black Body source", 5e4)
    cross_section_pl_index: float = _field(
        "Power-law index for the frequency dependence of the photoionization cross section",
        2.8,
    )
    sed_table_file: Path = _field("Path to the SED table file", Path(""))

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.sed_table_file = Path(self.sed_table_file)


@dataclass
class PhotoParameters(YmlParameters):
    """Parameters governing photoionization tables

    The following attributes are populated by the registered subclasses:
    - 'source_type_model' -> source_type [SourceTypeParameters]
    """

    SECTION: ClassVar[str] = "Photo"

    minlogtau: float = _field("Minimum optical depth for tables", -20)
    maxlogtau: float = _field("Maximum optical depth for tables", 4)
    num_tau: int = _field("Number of table points", 2000)
    grey: bool = _field(
        "Whether or not to use grey opacity (i.e. cross-section is frequency-independent)",
        False,
    )
    compute_heating_rates: bool = _field(
        "Whether to compute heating rates arrays", False
    )
    source_type: SourceTypeParameters = field(default_factory=BlackBodySourceParameters)

    @classmethod
    def from_dict(cls, yml: dict[str, Any], warn: bool = False) -> PhotoParameters:
        source_type = SourceTypeParameters.from_model(yml, "blackbody", warn)
        return super().from_dict({**yml, "source_type": source_type}, warn)

    @property
    def source_type_model(self) -> str:
        return self.source_type.MODEL


class ClumpingParameters(_ModelRegistry["ClumpingParameters"]):
    """Clumping model parameters"""

    _registry: ClassVar[dict[str, type[ClumpingParameters]]] = {}
    KEY: ClassVar[str] = "clumping_model"


@dataclass
class ConstantClumpingParameters(ClumpingParameters):
    MODEL: ClassVar[str] = "constant"

    value: float = _field("Clumping factor for the constant model", 5.0)


@dataclass
class RedshiftClumpingParameters(ClumpingParameters):
    MODEL: ClassVar[str] = "redshift"


@dataclass
class DensityClumpingParameters(ClumpingParameters):
    MODEL: ClassVar[str] = "density"


@dataclass
class StochasticClumpingParameters(ClumpingParameters):
    MODEL: ClassVar[str] = "stochastic"


class MfpParameters(_ModelRegistry["MfpParameters"]):
    """Mean-free-path model parameters"""

    _registry: ClassVar[dict[str, type[MfpParameters]]] = {}
    KEY: ClassVar[str] = "mfp_model"


@dataclass
class ConstantMfpParameters(MfpParameters):
    MODEL: ClassVar[str] = "constant"

    R_max_cMpc: float = _field(
        "Maximum comoving distance for photons from source", 15.0
    )


@dataclass
class Choudhury09MfpParameters(MfpParameters):
    MODEL: ClassVar[str] = "Choudhury09"

    A_mfp: float = _field(
        "Free parameter for the Choudhury09 mean-free-path model in cMpc units", 210.0
    )
    eta_mfp: float = _field(
        "Spectral index of the Choudhury09 mean-free-path model redshift evolution",
        -9.0,
    )


@dataclass
class Worseck14MfpParameters(MfpParameters):
    MODEL: ClassVar[str] = "Worseck14"

    A_mfp: float = _field(
        "Free parameter for the Worseck14 mean-free-path model in cMpc units", 210.0
    )
    eta_mfp: float = _field(
        "Spectral index of the Worseck14 mean-free-path model redshift evolution",
        -9.0,
    )
    eta1_mfp: float = _field("Parameter for the modification to the Worseck14 fit", 9.0)
    z1_mfp: float = _field("Parameter for the modification to the Worseck14 fit", 6.0)


@dataclass
class SinksParameters(YmlParameters):
    """Parameters for sinks

    The following attributes are populated by the registered subclasses:
    - 'clumping_model' -> clumping [ClumpingParameters]
    - 'mfp_model' -> mfp [MfpParameters]
    """

    SECTION: ClassVar[str] = "Sinks"

    clumping: ClumpingParameters = field(default_factory=ConstantClumpingParameters)
    mfp: MfpParameters = field(default_factory=ConstantMfpParameters)

    @classmethod
    def from_dict(cls, yml: dict[str, Any], warn: bool = False) -> SinksParameters:
        clumping = ClumpingParameters.from_model(yml, "constant", warn)
        mfp = MfpParameters.from_model(yml, "constant", warn)
        return super().from_dict({**yml, "clumping": clumping, "mfp": mfp}, warn)

    @property
    def clumping_model(self) -> str:
        return self.clumping.MODEL

    @property
    def mfp_model(self) -> str:
        return self.mfp.MODEL


class FstarParameters(_ModelRegistry["FstarParameters"]):
    """Stellar-to-halo mass relation parameters"""

    _registry: ClassVar[dict[str, type[FstarParameters]]] = {}
    KEY: ClassVar[str] = "fstar_model"


@dataclass
class FgammaSourceParameters(FstarParameters):
    """Classical mass-independent stellar-to-halo mass relation"""

    MODEL: ClassVar[str] = "fgamma"

    f0: float = _field("Constant stellar-to-halo mass fraction", 0.02)
    fgamma_hm: float = _field(
        "Efficiency High-Mass Atomically Cooling Halo (HMACH)", 30.0
    )
    fgamma_lm: float = _field(
        "Efficiency Low-Mass Atomically Cooling Halo (LMACH)", 0.0
    )


@dataclass
class DplSourceParameters(FstarParameters):
    """Double power law stellar-to-halo mass relation (Schneider, Giri, Mirocha '21)."""

    MODEL: ClassVar[str] = "dpl"

    f0: float = _field("Amplitude", 0.02)
    Mt: float = _field("Turn over mass", 1e10)
    Mp: float = _field("Second turn over mass", 1e10)
    g1: float = _field("First Mp exponent", -0.3)
    g2: float = _field("Second Mp exponent", -0.3)
    g3: float = _field("First Mt exponent", 0.0)
    g4: float = _field("Second Mt exponent", -0.0)

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        if self.g3 < 0 or self.g4 > 0:
            raise ValueError(
                f"Invalid DPL parameters: g3 = {self.g3} must be >= 0 and g4 = {self.g4} must be <= 0."
            )


@dataclass
class MuvSourceParameters(FstarParameters):
    """Scatter in the absolute magnitude with sigma_UV = a_s*log10(Mhalo)+b_s. This extends Gelli+ 2024"""

    MODEL: ClassVar[str] = "Muv"

    f0: float = _field("Amplitude", 0.02)
    Mt: float = _field("Turn over mass", 1e10)
    Mp: float = _field("Second turn over mass", 1e10)
    g1: float = _field("First Mp exponent", -0.3)
    g2: float = _field("Second Mp exponent", -0.3)
    g3: float = _field("First Mt exponent", 0.0)
    g4: float = _field("Second Mt exponent", -0.0)
    a_s: float = _field(
        "Slope of the linear relation between log mass and sigma_UV for the Muv-scatter",
        -1.0 / 3.0,
    )
    b_s: float = _field(
        "Intercept of the linear relation between log mass sigma_UV for the Muv-scatter",
        9.0 / 2.0,
    )

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        if self.g3 < 0 or self.g4 > 0:
            raise ValueError(
                f"Invalid Muv parameters: g3 = {self.g3} must be >= 0 and g4 = {self.g4} must be <= 0."
            )
        if self.a_s >= 0:
            raise ValueError(f"Invalid Muv parameters: a_s = {self.a_s} must be < 0.")
        if self.b_s <= 0:
            raise ValueError(f"Invalid Muv parameters: b_s = {self.b_s} must be > 0.")


class FescParameters(_ModelRegistry["FescParameters"]):
    """Escaping photon fraction parameters"""

    _registry: ClassVar[dict[str, type[FescParameters]]] = {}
    KEY: ClassVar[str] = "fesc_model"


@dataclass
class ConstantFescParameters(FescParameters):
    """Mass-independent escape fraction"""

    MODEL: ClassVar[str] = "constant"

    f0_esc: float = _field("Escape fraction parameter", 0.1)


@dataclass
class PowerFescParameters(FescParameters):
    """Power law mass-dependent escape fraction"""

    MODEL: ClassVar[str] = "power"

    f0_esc: float = _field("Escape fraction parameter", 0.1)
    Mp_esc: float = _field("Escape fraction parameter", 1e10)
    al_esc: float = _field("Escape fraction parameter", -0.3)


@dataclass
class PowerObsFescParameters(FescParameters):
    """Power law mass-dependent escape fraction, fitted to data that uses stellar mass"""

    MODEL: ClassVar[str] = "power_obs"

    f0_esc: float = _field("Escape fraction parameter", 0.1)
    Mp_esc: float = _field("Escape fraction parameter", 1e10)
    al_esc: float = _field(
        "Power index of the mass dependence for the escape fraction parameter", -0.3
    )
    al_esc_z: float = _field(
        "Power index of the mass dependence for escape fraction parameter", 1.0
    )
    zp_esc: float = _field("Zero point of the escape fraction parameter", 0.0)


@dataclass
class ThesanFescParameters(FescParameters):
    """Thesan escape fraction model"""

    MODEL: ClassVar[str] = "thesan"

    # TODO: path to Thesan escape fraction model file
    tables_dir: PathType = _field(
        "Path to directory with the Thesan escape fraction tables", Path("")
    )

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.tables_dir = Path(self.tables_dir)


class AccretionParameters(_ModelRegistry["AccretionParameters"]):
    """Accretion model parameters"""

    _registry: ClassVar[dict[str, type[AccretionParameters]]] = {}
    KEY: ClassVar[str] = "accretion_model"


@dataclass
class ConstantAccretionParameters(AccretionParameters):
    MODEL: ClassVar[str] = "constant"


@dataclass
class ExpAccretionParameters(AccretionParameters):
    MODEL: ClassVar[str] = "exp"

    # TODO: remove
    bursty_sfr: Literal["no", "instant", "integrate"] = _field(
        "Bursty star-formation model", "no"
    )

    alpha_h: float = _field("accretion rate parameter (see Schneider+21)", 0.79)
    beta1: float = _field(
        "Index power-low of the bursty star-formation model mass relation", 0.1
    )
    beta2: float = _field(
        "Index power-low of the bursty star-formation model time relation", 1.5
    )
    tB0: float = _field("Bursty star-formation time-scale at z=0", 200.0)
    tQ_frac: float = _field("Fraction of the quiescent time-scale", 1.6)
    z0: float = _field("Reference redshift for the bursty star-formation model", 30.0)
    t_rnd: float = _field(
        "Randomize the time-scale of the bursty star-formation model (std for "
        "N~(t_start, t_rnd))",
        0,
    )


@dataclass
class SourcesParameters(YmlParameters):
    """Parameters for sources

    The following attributes are populated by the registered subclasses:
    - 'fstar_model' -> fstar [FstarParameters]
    - 'fesc_model' -> fesc [FescParameters]
    - 'accretion_model' -> accretion [AccretionParameters]
    """

    SECTION: ClassVar[str] = "Sources"

    num_ion: int = _field("Number of ionizing photons per baryon in stars", 2000)
    fstar: FstarParameters = field(default_factory=FgammaSourceParameters)
    fesc: FescParameters = field(default_factory=ConstantFescParameters)
    accretion: AccretionParameters = field(default_factory=ConstantAccretionParameters)
    seed: int | None = _field(
        "Seed for the stochastic source models (None: different on every run)", None
    )

    @classmethod
    def from_dict(cls, yml: dict[str, Any], warn: bool = False) -> SourcesParameters:
        fstar = FstarParameters.from_model(yml, "fgamma", warn)
        fesc = FescParameters.from_model(yml, "constant", warn)
        accretion = AccretionParameters.from_model(yml, "constant", warn)
        return super().from_dict(
            {**yml, "fstar": fstar, "fesc": fesc, "accretion": accretion}, warn
        )

    @property
    def fstar_model(self) -> str:
        return self.fstar.MODEL

    @property
    def fesc_model(self) -> str:
        return self.fesc.MODEL

    @property
    def accretion_model(self) -> str:
        return self.accretion.MODEL


@dataclass
class SimulationParameters:
    """Aggregates every parameter section into a single, serializable object"""

    io: InputOutputParameters
    grid: GridParameters
    raytracing: RaytracingParameters
    domain_decomposition: DomainDecompositionParameters
    material: MaterialParameters
    chemistry: ChemistryParameters
    cosmology: CosmologyParameters
    photo: PhotoParameters
    sinks: SinksParameters
    sources: SourcesParameters

    @classmethod
    def from_file(cls, file: PathType) -> SimulationParameters:
        """Load every parameter section from a single YAML file."""
        yml = YmlParameters.load_yaml(file)
        hints = get_type_hints(cls)
        param_types = {f.name: hints[f.name] for f in fields(cls)}

        return cls(
            **{
                name: param.from_dict(yml.get(param.SECTION, {}))
                for name, param in param_types.items()
            }
        )

    def to_yaml_file(self, file: PathType) -> None:
        """Write every section's current values to a single YAML file."""
        blocks = [getattr(self, f.name).to_yaml_block() for f in fields(self)]
        with open(file, "w") as fh:
            fh.write(
                "# pyC2Ray parameters file -- generated, edit values as needed.\n\n"
            )
            fh.write("\n\n".join(blocks))
            fh.write("\n")
