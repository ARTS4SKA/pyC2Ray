from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from dataclasses import KW_ONLY, InitVar, dataclass, field, fields
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


# Configure YML to read scientific notation as floats rather than strings
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _Loader.YML_REGEX, list("-+0123456789.")
)

PathType = str | os.PathLike
OptFloat = float | None
OptStr = str | None
ParametersType = TypeVar("ParametersType", bound="YmlParameters")

logger = logging.getLogger(__name__)


def _field(default: Any, doc: str) -> Any:
    """Shorthand for a dataclass field carrying a human-readable description.

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
        cls: type[ParametersType], yml: dict[str, Any], warn: bool = True
    ) -> ParametersType:
        """Instantiate from a dict, optionally warning about missing keys."""
        keys = {f.name for f in fields(cls)}
        if warn:
            missing_keys = keys - yml.keys()
            if missing_keys:
                keys_str = ", ".join(missing_keys)
                conf_str = getattr(cls, "SECTION", cls.__name__)
                logger.warning(
                    f"Missing key(s) for '{conf_str}' configuration: {keys_str}"
                )
        return cls(**{k: v for k, v in yml.items() if k in keys}, _build=_factory)

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

    def to_yaml_block(self, *, header: bool = True) -> str:
        """Render this section's current values as a commented YAML block.

        Parameters
        ----------
        header :
            Whether to include the `SECTION:` line at the top of the block.
        """
        lines = [f"{self.SECTION}:"] if header else []
        for f in fields(self):
            doc = f.metadata.get("doc")
            if doc:
                lines.append(f"  # {doc}")
            dumped = yaml.safe_dump(
                {f.name: getattr(self, f.name)},
                default_flow_style=False,
                sort_keys=False,
            )
            lines.append("  " + dumped.strip())
        return "\n".join(lines)


RegistryKind = TypeVar("RegistryKind", bound="_KindRegistry")


class _KindRegistry(YmlParameters, Generic[RegistryKind]):
    """Base for a group of mutually-exclusive parameter sets selected by a
    string discriminator (e.g. `fstar_kind`, `mfp_model`).

    Each concrete "kind" subclasses defines sets the `KIND` variable and
    are automatically registered to the parameter category's register via
    `__init_subclass__`.

    Not a `@dataclass` itself and declares no fields, so it can be freely
    subclassed by leaf dataclasses without affecting their field ordering.
    """

    KIND: ClassVar[str]
    _registry: ClassVar[dict[str, type[RegistryKind]]]

    def __init_subclass__(cls: type[RegistryKind], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "KIND"):
            cls._registry[cls.KIND] = cls

    @classmethod
    def from_kind(
        cls, kind: str, yml: dict[str, Any], warn: bool = True
    ) -> RegistryKind:
        """Instantiate whichever registered subclass matches `kind`.

        Reads only the keys relevant to that subclass out of `yml`.
        It is expected to be the *whole* enclosing section's dict (siblings
        belonging to other kinds are silently ignored, not warned about).
        """
        try:
            subclass = cls._registry[kind]
        except KeyError:
            raise ValueError(
                f"{kind!r} is not implemented for {cls.__name__}. "
                f"Choose from {sorted(cls._registry)!r}."
            )
        return subclass.from_dict(yml, warn=warn)


@dataclass()
class OutputParameters(YmlParameters):
    """Output setup"""

    SECTION: ClassVar[str] = "Output"

    results_basename: str = _field(
        "results", "Directory where results and log files are stored"
    )
    inputs_basename: OptStr = _field(None, "Directory where input files are stored")
    sources_basename: OptStr = _field(None, "Basename of the sources file")
    density_basename: OptStr = _field(None, "Basename of the density file")
    logfile: str = _field("pyC2Ray.log", "Name of the log file to write")


@dataclass
class GridParameters(YmlParameters):
    """Parameters to set up the simulation volume"""

    SECTION: ClassVar[str] = "Grid"

    boxsize: float = _field(10.0, "Box size in comoving Mpc")
    meshsize: int = _field(256, "Side of the mesh grid")
    gpu: bool = _field(False, "Use GPU acceleration")
    mpi: bool = _field(False, "Use MPI parallelization")
    resume: bool = _field(False, "Resume a simulation")


@dataclass
class RaytracingParameters(YmlParameters):
    """ASORA and C2Ray (legacy) raytracing parameters"""

    SECTION: ClassVar[str] = "Raytracing"

    loss_fraction: float = _field(1e-2, "Photon loss fraction for the subbox algorithm")
    subboxsize: int = _field(128, "Size increase of subboxes around sources")
    max_subbox: int = _field(1000, "Maximum subbox size for the subbox algorithm")
    source_batch_size: int = _field(1, "Number of sources to be processed in parallel")
    convergence_fraction: float = _field(
        1.0e-4, "Which fraction of the cells can be left unconverged"
    )


@dataclass
class DomainDecompositionParameters(YmlParameters):
    """Parameters for domain decomposition and source grouping."""

    SECTION: ClassVar[str] = "DomainDecomposition"

    enabled: bool = _field(False, "Enable domain decomposition/source grouping")
    grouping_algorithm: Literal["morton"] = _field(
        "morton", "Source grouping/domain decomposition algorithm"
    )
    max_num_sources_per_group: int = _field(
        1000, "Maximum number of sources in one source group"
    )
    # TODO: morton_bits is a morton specific parameter, should be moved to a morton grouping parameter class
    morton_bits: int = _field(10, "Number of bits per dimension for Morton ordering")
    max_memory_cost_per_group: float = _field(
        50.0e9, "Maximum allowed memory cost per group in bytes"
    )

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        if self.max_num_sources_per_group <= 0:
            raise ValueError(
                f"max_num_sources_per_group must be a positive integer. "
                f"Provided value is {self.max_num_sources_per_group}."
            )
        if self.morton_bits <= 0:
            raise ValueError(
                f"morton_bits must be a positive integer. "
                f"Provided value is {self.morton_bits}."
            )
        if self.max_memory_cost_per_group <= 0:
            raise ValueError(
                f"max_memory_cost_per_group must be a positive number. "
                f"Provided value is {self.max_memory_cost_per_group}."
            )


@dataclass
class MaterialParameters(YmlParameters):
    """Properties of physical quantities in the simulation volume"""

    SECTION: ClassVar[str] = "Material"

    temp0: float = _field(1e4, "Initial Temperature of the grid")
    xHII: float = _field(1.2e-3, "Initial Ionized fraction of the grid")
    avg_dens: float = _field(1.87e-7, "Constant average density, comoving value")


@dataclass
class AbundancesParameters(YmlParameters):
    """Element abundances"""

    SECTION: ClassVar[str] = "Abundances"

    abu_h: float = _field(0.926, "Hydrogen Abundance")
    abu_he: float = _field(0.074, "Helium Abundance")
    abu_c: float = _field(7.1e-7, "Carbon Abundance")

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.mean_molecular = self.abu_h + 4.0 * self.abu_he


@dataclass
class CGSParameters(YmlParameters):
    """Miscellaneous physical constants"""

    # TODO: we should move these values to constants.

    SECTION: ClassVar[str] = "CGS"

    albpow: float = _field(-0.7, "Hydrogen recombination parameter (power law index)")
    b0_HI: float = _field(
        2.59e-13, "Hydrogen recombination parameter (value at 10^4 K)"
    )
    alcpow: float = _field(-0.67, "Helium0 recombination parameter (power law index)")
    ion_energy_HI: float = _field(13.598, "Hydrogen I ionization energy (in eV)")
    ion_energy_HeI: float = _field(24.587, "Helium I ionization energy (in eV)")
    ion_energy_HeII: float = _field(54.416, "Helium II ionization energy (in eV)")
    xi_HI: float = _field(1.0, "Hydrogen collisional ionization parameter 1")
    f_HI: float = _field(0.83, "Hydrogen collisional ionization parameter 2")
    col_HI_fact: float = _field(1.3e-8, "Colf_HI factor")

    def __post_init__(self, _build: object) -> None:
        super().__post_init__(_build)
        self.col_HI = self.col_HI_fact * self.f_HI * self.xi_HI / self.ion_energy_HI**2
        self.temp_HI = self.ion_energy_HI * c.ev2k


@dataclass
class CosmologyParameters(YmlParameters):
    """Cosmological Parameters"""

    SECTION: ClassVar[str] = "Cosmology"

    cosmological: bool = _field(False, "Global flag to use cosmology")
    h: float = _field(0.7, "Reduced Hubble constant")
    Omega0: float = _field(0.27, "Omega matter t=0")
    Omega_B: float = _field(0.043, "Omega baryon t=0")
    cmbtemp: float = _field(2.726, "Temperature of CMB in Kelvin")
    zred_0: float = _field(9.0, "Initial redshift of the simulation")


@dataclass
class PhotoParameters(YmlParameters):
    """Parameters governing photoionization"""

    SECTION: ClassVar[str] = "Photo"

    sigma_HI_at_ion_freq: float = _field(
        6.30e-18, "HI cross section at its ionizing frequency (weighted by freq_factor)"
    )
    minlogtau: float = _field(-20, "Minimum optical depth for tables")
    maxlogtau: float = _field(4, "Maximum optical depth for tables")
    num_tau: int = _field(20000, "Number of table points")
    grey: bool = _field(
        False,
        "Whether or not to use grey opacity (i.e. cross-section is frequency-independent)",
    )
    # TODO: expand this parameter class for different source types
    source_type: Literal["blackbody", "powerlaw", "Zackrisson11"] = _field(
        "blackbody", "Type of source to use"
    )
    compute_heating_rates: bool = _field(
        False, "Whether to compute heating rates arrays"
    )
    sed_table_path: OptStr = _field(None, "Path to the SED table file")


@dataclass
class BlackBodySourceParameters(YmlParameters):
    """Parameters for Black Body source type"""

    SECTION: ClassVar[str] = "BlackBodySource"

    Teff: float = _field(5e4, "Effective temperature of Black Body source")
    cross_section_pl_index: float = _field(
        2.8,
        "Power-law index for the frequency dependence of the photoionization cross section",
    )


class ClumpingParameters(_KindRegistry["ClumpingParameters"]):
    """Clumping model parameters, selected by `clumping_model`."""

    _registry: ClassVar[dict[str, type[ClumpingParameters]]] = {}


@dataclass
class ConstantClumpingParameters(ClumpingParameters):
    KIND: ClassVar[str] = "constant"

    value: float = _field(5.0, "Clumping factor for the constant model")


@dataclass
class RedshiftClumpingParameters(ClumpingParameters):
    KIND: ClassVar[str] = "redshift"


@dataclass
class DensityClumpingParameters(ClumpingParameters):
    KIND: ClassVar[str] = "density"


@dataclass
class StochasticClumpingParameters(ClumpingParameters):
    KIND: ClassVar[str] = "stochastic"


class MfpParameters(_KindRegistry["MfpParameters"]):
    """Mean-free-path model parameters, selected by `mfp_model`."""

    _registry: ClassVar[dict[str, type[MfpParameters]]] = {}


@dataclass
class ConstantMfpParameters(MfpParameters):
    KIND: ClassVar[str] = "constant"

    R_max_cMpc: float = _field(
        15.0, "Maximum comoving distance for photons from source"
    )


@dataclass
class Choudhury09MfpParameters(MfpParameters):
    KIND: ClassVar[str] = "Choudhury09"

    A_mfp: OptFloat = _field(
        None, "Free parameter for the Choudhury09 mean-free-path model in cMpc units"
    )
    eta_mfp: OptFloat = _field(
        None,
        "Spectral index of the Choudhury09 mean-free-path model redshift evolution",
    )


@dataclass
class Worseck14MfpParameters(MfpParameters):
    KIND: ClassVar[str] = "Worseck14"

    A_mfp: float = _field(
        210.0, "Free parameter for the Worseck14 mean-free-path model in cMpc units"
    )
    eta_mfp: float = _field(
        -9.0,
        "Spectral index of the Worseck14 mean-free-path model redshift evolution",
    )
    eta1_mfp: float = _field(9.0, "Parameter for the modification to the Worseck14 fit")
    z1_mfp: float = _field(6.0, "Parameter for the modification to the Worseck14 fit")


@dataclass
class SinksParameters(YmlParameters):
    """Parameters for sinks.

    The attributes are populated by the registered subclasses based
    on the values of the following keys in the input dictionary:
    - 'clumping_model' -> clumping [ClumpingParameters]
    - 'mfp_model' -> mfp [MfpParameters]
    """

    SECTION: ClassVar[str] = "Sinks"

    clumping: ClumpingParameters = field(default_factory=ConstantClumpingParameters)
    mfp: MfpParameters = field(default_factory=ConstantMfpParameters)

    @classmethod
    def from_dict(cls, yml: dict[str, Any], warn: bool = True) -> SinksParameters:
        return cls(
            clumping=ClumpingParameters.from_kind(
                yml.get("clumping_model", "constant"), yml, warn
            ),
            mfp=MfpParameters.from_kind(yml.get("mfp_model", "constant"), yml, warn),
            _build=_factory,
        )

    @property
    def clumping_model(self) -> str:
        return self.clumping.KIND

    @property
    def mfp_model(self) -> str:
        return self.mfp.KIND

    def to_yaml_block(self, *, header: bool = True) -> str:
        lines = [f"{self.SECTION}:"] if header else []
        for doc, kind_key, obj in (
            ("Clumping model", "clumping_model", self.clumping),
            ("Mean-free-path model", "mfp_model", self.mfp),
        ):
            lines.append(f"  # {doc}")
            dumped = yaml.safe_dump(
                {kind_key: obj.KIND}, default_flow_style=False, sort_keys=False
            )
            lines.append("  " + dumped.strip())
            lines.extend(obj.to_yaml_block(header=False).splitlines())
        return "\n".join(lines)


class FstarParameters(_KindRegistry["FstarParameters"]):
    """Stellar-to-halo mass relation parameters, selected by `fstar_kind`."""

    _registry: ClassVar[dict[str, type[FstarParameters]]] = {}


@dataclass
class FgammaSourceParameters(FstarParameters):
    """Classical mass-independent stellar-to-halo mass relation."""

    KIND: ClassVar[str] = "fgamma"

    fgamma_hm: OptFloat = _field(
        None, "Efficiency High-Mass Atomically Cooling Halo (HMACH)"
    )
    fgamma_lm: OptFloat = _field(
        None, "Efficiency Low-Mass Atomically Cooling Halo (LMACH)"
    )


@dataclass
class DplSourceParameters(FstarParameters):
    """Double power law stellar-to-halo mass relation (Schneider, Giri, Mirocha '21)."""

    KIND: ClassVar[str] = "dpl"

    Nion: OptFloat = _field(None, "Double power law parameter")
    f0: OptFloat = _field(None, "Double power law parameter")
    Mt: OptFloat = _field(None, "Double power law parameter")
    Mp: OptFloat = _field(None, "Double power law parameter")
    g1: OptFloat = _field(None, "Double power law parameter")
    g2: OptFloat = _field(None, "Double power law parameter")
    g3: OptFloat = _field(None, "Double power law parameter")
    g4: OptFloat = _field(None, "Double power law parameter")


@dataclass
class LognormSourceParameters(FstarParameters):
    """Stochastic stellar-to-halo mass relation with lognorm distribution and
    std ~Mhalo^(-1/3)."""

    KIND: ClassVar[str] = "lognorm"


@dataclass
class MuvSourceParameters(FstarParameters):
    """Scatter in the absolute magnitude with std_UV~a_s*log10(Mhalo)+b_s
    (see Gelli+24)."""

    KIND: ClassVar[str] = "Muv"

    a_s: OptFloat = _field(
        None, "Free parameter for the Muv-scatter in the fstar model"
    )
    b_s: OptFloat = _field(
        None, "Free parameter for the Muv-scatter in the fstar model"
    )


class FescParameters(_KindRegistry["FescParameters"]):
    """Escaping photon fraction parameters, selected by `fesc_model`."""

    _registry: ClassVar[dict[str, type[FescParameters]]] = {}


@dataclass
class ConstantFescParameters(FescParameters):
    """Mass-independent escape fraction."""

    KIND: ClassVar[str] = "constant"

    f0_esc: OptFloat = _field(None, "Escape fraction parameter")


@dataclass
class PowerFescParameters(FescParameters):
    """Power law mass-dependent escape fraction."""

    KIND: ClassVar[str] = "power"

    f0_esc: OptFloat = _field(None, "Escape fraction parameter")
    Mp_esc: OptFloat = _field(None, "Escape fraction parameter")
    al_esc: OptFloat = _field(None, "Escape fraction parameter")


@dataclass
class PowerObsFescParameters(FescParameters):
    """Power law mass-dependent escape fraction, fitted to data that uses stellar mass."""

    KIND: ClassVar[str] = "power_obs"


@dataclass
class Gelli24FescParameters(FescParameters):
    """UV-dependent escape fraction model (Gelli+24)."""

    KIND: ClassVar[str] = "Gelli24"


@dataclass
class ThesanFescParameters(FescParameters):
    """Thesan escape fraction model."""

    KIND: ClassVar[str] = "Thesan"


class AccretionParameters(_KindRegistry["AccretionParameters"]):
    """Accretion model parameters, selected by `accretion_model`."""

    _registry: ClassVar[dict[str, type[AccretionParameters]]] = {}


@dataclass
class ConstantAccretionParameters(AccretionParameters):
    KIND: ClassVar[str] = "constant"


@dataclass
class ExpAccretionParameters(AccretionParameters):
    KIND: ClassVar[str] = "exp"

    alpha_h: OptFloat = _field(None, "accretion rate parameter (see Schneider+21)")
    bursty_sfr: Literal["no", "instant", "integrate"] = _field(
        "no", "Bursty star-formation model"
    )
    beta1: OptFloat = _field(
        None, "Index power-low of the bursty star-formation model mass relation"
    )
    beta2: OptFloat = _field(
        None, "Index power-low of the bursty star-formation model time relation"
    )
    tB0: OptFloat = _field(None, "Bursty star-formation time-scale at z=0")
    tQ_frac: OptFloat = _field(None, "Fraction of the quiescent time-scale")
    z0: OptFloat = _field(
        None, "Reference redshift for the bursty star-formation model"
    )
    t_rnd: OptFloat = _field(
        None,
        "Randomize the time-scale of the bursty star-formation model (std for "
        "N~(t_start, t_rnd))",
    )


@dataclass
class SourcesParameters(YmlParameters):
    """Parameters for sources.

    The attributes are populated by the registered subclasses based
    on the values of the following keys in the input dictionary:
    - 'fstar_kind' -> fstar [FstarParameters]
    - 'fesc_model' -> fesc [FescParameters]
    - 'accretion_model' -> accretion [AccretionParameters]
    """

    SECTION: ClassVar[str] = "Sources"

    fstar: FstarParameters = field(default_factory=FgammaSourceParameters)
    fesc: FescParameters = field(default_factory=ConstantFescParameters)
    accretion: AccretionParameters = field(default_factory=ConstantAccretionParameters)
    ts: OptFloat = _field(None, "TODO: add description")

    @classmethod
    def from_dict(cls, yml: dict[str, Any], warn: bool = True) -> SourcesParameters:
        return cls(
            fstar=FstarParameters.from_kind(yml.get("fstar_kind", "fgamma"), yml, warn),
            fesc=FescParameters.from_kind(yml.get("fesc_model", "constant"), yml, warn),
            accretion=AccretionParameters.from_kind(
                yml.get("accretion_model", "constant"), yml, warn
            ),
            ts=yml.get("ts"),
            _build=_factory,
        )

    @property
    def fstar_kind(self) -> str:
        return self.fstar.KIND

    @property
    def fesc_model(self) -> str:
        return self.fesc.KIND

    @property
    def accretion_model(self) -> str:
        return self.accretion.KIND

    def to_yaml_block(self, *, header: bool = True) -> str:
        lines = [f"{self.SECTION}:"] if header else []
        for doc, kind_key, obj in (
            ("Stellar-to-halo mass relation", "fstar_kind", self.fstar),
            ("Escaping photon fraction model", "fesc_model", self.fesc),
            ("Accretion model", "accretion_model", self.accretion),
        ):
            lines.append(f"  # {doc}")
            dumped = yaml.safe_dump(
                {kind_key: obj.KIND}, default_flow_style=False, sort_keys=False
            )
            lines.append("  " + dumped.strip())
            lines.extend(obj.to_yaml_block(header=False).splitlines())
        for f in fields(self):
            if f.name in ("fstar", "fesc", "accretion"):
                continue
            doc = f.metadata.get("doc", "")
            if doc:
                lines.append(f"  # {doc}")
            dumped = yaml.safe_dump(
                {f.name: getattr(self, f.name)},
                default_flow_style=False,
                sort_keys=False,
            )
            lines.append("  " + dumped.strip())
        return "\n".join(lines)


# All concrete parameter sections. Used by `generate_parameter_file` (and
# `SimulationParameters`) to produce/read a complete parameters.yml.
ALL_PARAMETER_CLASSES: tuple[type[YmlParameters], ...] = (
    OutputParameters,
    GridParameters,
    RaytracingParameters,
    DomainDecompositionParameters,
    MaterialParameters,
    AbundancesParameters,
    CGSParameters,
    CosmologyParameters,
    BlackBodySourceParameters,
    PhotoParameters,
    SinksParameters,
    SourcesParameters,
)


def generate_parameter_file(
    file: PathType,
    sections: Iterable[type[YmlParameters] | YmlParameters] = ALL_PARAMETER_CLASSES,
) -> None:
    """Write a fully-commented YAML parameters file.

    Each entry in `sections` may be a `YmlParameters` subclass, in which case
    its default values are written (useful to generate a starting template
    for a new simulation), or an already-constructed instance, in which case
    its current values are written (useful to snapshot the exact
    configuration used for a run).

    Parameters
    ----------
    file :
        Path of the YAML file to write.
    sections :
        `YmlParameters` subclasses and/or instances to write, in order.
        Defaults to every built-in section.
    """
    blocks = [
        (section() if isinstance(section, type) else section).to_yaml_block()
        for section in sections
    ]
    with open(file, "w") as fh:
        fh.write("# pyC2Ray parameters file -- generated, edit values as needed.\n\n")
        fh.write("\n\n".join(blocks))
        fh.write("\n")


def _param_field(param: type[YmlParameters]) -> Any:
    """Shorthand for a SimulationParameters field.

    The YmlParameters instances can only be created via the factory methods,
    so `field(default_factory=...)` can't be used directly.
    This helper function to create a field with a default factory that calls the factory method.
    """
    return field(default_factory=lambda: param.from_dict({}, warn=False))


@dataclass
class SimulationParameters:
    """Aggregates every parameter section into a single, serializable object."""

    output: OutputParameters = _param_field(OutputParameters)
    grid: GridParameters = _param_field(GridParameters)
    raytracing: RaytracingParameters = _param_field(RaytracingParameters)
    domain_decomposition: DomainDecompositionParameters = _param_field(
        DomainDecompositionParameters
    )
    material: MaterialParameters = _param_field(MaterialParameters)
    abundances: AbundancesParameters = _param_field(AbundancesParameters)
    cgs: CGSParameters = _param_field(CGSParameters)
    cosmology: CosmologyParameters = _param_field(CosmologyParameters)
    photo: PhotoParameters = _param_field(PhotoParameters)
    blackbody: BlackBodySourceParameters = _param_field(BlackBodySourceParameters)
    sinks: SinksParameters = _param_field(SinksParameters)
    sources: SourcesParameters = _param_field(SourcesParameters)

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
        generate_parameter_file(
            file,
            sections=(
                self.output,
                self.grid,
                self.raytracing,
                self.domain_decomposition,
                self.material,
                self.cgs,
                self.cosmology,
                self.abundances,
                self.photo,
                self.sinks,
                self.blackbody,
                self.sources,
            ),
        )
