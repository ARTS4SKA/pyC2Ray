from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import pytest

import pyc2ray.constants as c
from pyc2ray.parameters import (
    AbundancesParameters,
    BlackBodySourceParameters,
    CGSParameters,
    Choudhury09MfpParameters,
    ConstantAccretionParameters,
    ConstantClumpingParameters,
    ConstantFescParameters,
    ConstantMfpParameters,
    CosmologyParameters,
    DensityClumpingParameters,
    DomainDecompositionParameters,
    DplSourceParameters,
    ExpAccretionParameters,
    FgammaSourceParameters,
    Gelli24FescParameters,
    GridParameters,
    LognormSourceParameters,
    MaterialParameters,
    MuvSourceParameters,
    OutputParameters,
    PhotoParameters,
    PowerFescParameters,
    PowerObsFescParameters,
    RaytracingParameters,
    RedshiftClumpingParameters,
    SimulationParameters,
    SinksParameters,
    SourcesParameters,
    StochasticClumpingParameters,
    ThesanFescParameters,
    Worseck14MfpParameters,
    YmlParameters,
    _factory,
    _KindRegistry,
    generate_parameter_file,
)


class TestYmlParameters:
    @dataclass
    class MockParameters(YmlParameters):
        SECTION: ClassVar[str] = "Mock"

        param1: int = field(default=0, metadata={"doc": "An integer parameter"})
        param2: float = field(default=1.0, metadata={"doc": "A float parameter"})
        param3: str = field(default="default", metadata={"doc": "A string parameter"})
        param4: Literal["a", "b", "c"] = field(
            default="a", metadata={"doc": "A parameter with limited choices"}
        )
        optparam: str | None = field(
            default=None, metadata={"doc": "An optional parameter"}
        )

    def test_default_no_init(self):
        with pytest.raises(TypeError):
            TestYmlParameters.MockParameters()

    def test_default(self):
        obj = TestYmlParameters.MockParameters.from_dict({})

        assert obj.param1 == 0
        assert obj.param2 == pytest.approx(1.0)
        assert obj.param3 == "default"
        assert obj.param4 == "a"
        assert obj.optparam is None

    def test_from_dict_with_warning_default_missing_keys(self, caplog):
        obj = TestYmlParameters.MockParameters.from_dict(
            {
                "param1": 10,
                "param2": 3.14,
                "param3": "custom",
                "param5": "ignored",
            }
        )

        assert obj.param1 == 10
        assert obj.param2 == pytest.approx(3.14)
        assert obj.param3 == "custom"
        assert obj.param4 == "a"
        assert obj.optparam is None

        # Warning emitted
        assert not hasattr(obj, "param5")
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING

    def test_from_dict_without_warning_default_missing_keys(self, caplog):
        obj = TestYmlParameters.MockParameters.from_dict(
            {
                "param1": 10,
                "param2": 3.14,
                "param3": "custom",
                "param5": "ignored",
            },
            warn=False,
        )

        assert obj.param1 == 10
        assert obj.param2 == pytest.approx(3.14)
        assert obj.param3 == "custom"
        assert obj.param4 == "a"
        assert obj.optparam is None

        # Warning not emitted
        assert not hasattr(obj, "param5")
        assert len(caplog.records) == 0

    def test_validate_literas(self):
        # Valid value
        for value in ["a", "b", "c"]:
            obj = TestYmlParameters.MockParameters.from_dict({"param4": value})
            assert obj.param4 == value

        # Invalid value
        with pytest.raises(ValueError):
            TestYmlParameters.MockParameters.from_dict({"param4": "invalid"})

    def test_to_yaml_block(self):
        obj = TestYmlParameters.MockParameters.from_dict(
            {
                "param1": 10,
                "param2": 3.14,
                "param3": "custom",
                "param4": "b",
                "param5": "ignored",
            }
        )
        yaml_block = obj.to_yaml_block()

        expected_yaml_block = """
Mock:
  # An integer parameter
  param1: 10
  # A float parameter
  param2: 3.14
  # A string parameter
  param3: custom
  # A parameter with limited choices
  param4: b
  # An optional parameter
  optparam: null
""".strip()

        assert yaml_block == expected_yaml_block

    @pytest.fixture(scope="class")
    def params_file(self, data_dir: Path) -> Path:
        return data_dir / "parameters.yml"

    def test_from_file(self, params_file: Path):
        obj = TestYmlParameters.MockParameters.from_file(params_file)
        assert obj.param1 == 10
        assert obj.param2 == pytest.approx(3.14)
        assert obj.param3 == "custom"
        assert obj.param4 == "b"
        assert obj.optparam is None


class ModelParameters(_KindRegistry["ModelParameters"]):
    """Clumping model parameters, selected by `clumping_model`."""

    _registry: ClassVar[dict] = {}


@dataclass
class ConstantModelParameters(ModelParameters):
    KIND: ClassVar[str] = "constant"

    value: float = field(default=5.0, metadata={"doc": "A constant parameter"})


class TestKindRegistry:
    @dataclass
    class MockParameters(YmlParameters):
        SECTION: ClassVar[str] = "Mock"

        param1: int = field(default=0, metadata={"doc": "An integer parameter"})
        model: ModelParameters = field(default_factory=ModelParameters)

        @classmethod
        def from_dict(cls, yml, warn=True):
            return cls(
                param1=yml.get("param1", 0),
                model=ModelParameters.from_kind(
                    yml.get("model", "constant"), yml, warn
                ),
                _build=_factory,
            )

    def test_kind_registry(self):
        obj = TestKindRegistry.MockParameters.from_dict(
            {"param1": 10, "model": "constant", "value": 1.0}
        )

        assert obj.param1 == 10
        assert isinstance(obj.model, ConstantModelParameters)
        assert obj.model.value == pytest.approx(1.0)


class TestOutputParameters:
    def test_output_parameters(self):
        obj = OutputParameters.from_dict({})

        assert obj.results_basename == "results"
        assert obj.inputs_basename is None
        assert obj.sources_basename is None
        assert obj.density_basename is None
        assert obj.logfile == "pyC2Ray.log"


class TestGridParameters:
    def test_grid_parameters(self):
        obj = GridParameters.from_dict({})

        assert obj.boxsize == pytest.approx(10.0)
        assert obj.meshsize == 256
        assert not obj.gpu
        assert not obj.mpi
        assert not obj.resume


class TestRaytracingParameters:
    def test_raytracing_parameters(self):
        obj = RaytracingParameters.from_dict({})

        assert obj.loss_fraction == pytest.approx(1e-2)
        assert obj.subboxsize == 128
        assert obj.max_subbox == 1000
        assert obj.source_batch_size == 1
        assert obj.convergence_fraction == pytest.approx(1e-4)


class TestDomainDecompositionParameters:
    def test_domain_decomposition_rejects_unknown_grouping_algorithm(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"grouping_algorithm": "unknown"})

    def test_domain_decomposition_rejects_non_positive_max_num_sources_per_group(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"max_num_sources_per_group": -1})

    def test_domain_decomposition_rejects_non_positive_morton_bits(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"morton_bits": -1})

    def test_domain_decomposition_rejects_non_positive_max_memory_cost_per_group(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"max_memory_cost_per_group": -1})

    def test_domain_decomposition_parameters(self):
        obj = DomainDecompositionParameters.from_dict({})

        assert not obj.enabled
        assert obj.grouping_algorithm == "morton"
        assert obj.max_num_sources_per_group == 1000
        assert obj.morton_bits == 10
        assert obj.max_memory_cost_per_group == pytest.approx(50.0e9)


class TestMaterialParameters:
    def test_material_parameters(self):
        obj = MaterialParameters.from_dict({})

        assert obj.temp0 == pytest.approx(1e4)
        assert obj.xHII == pytest.approx(1.2e-3)
        assert obj.avg_dens == pytest.approx(1.87e-7)


class TestAbundancesParameters:
    def test_abundances_parameters(self):
        obj = AbundancesParameters.from_dict({})

        assert obj.abu_h == pytest.approx(0.926)
        assert obj.abu_he == pytest.approx(0.074)
        assert obj.abu_c == pytest.approx(7.1e-7)
        assert obj.mean_molecular == pytest.approx(0.926 + 4.0 * 0.074)


class TestCGSParameters:
    def test_cgs_parameters(self):
        obj = CGSParameters.from_dict({})

        assert obj.albpow == pytest.approx(-0.7)
        assert obj.b0_HI == pytest.approx(2.59e-13)
        assert obj.alcpow == pytest.approx(-0.67)
        assert obj.ion_energy_HI == pytest.approx(13.598)
        assert obj.ion_energy_HeI == pytest.approx(24.587)
        assert obj.ion_energy_HeII == pytest.approx(54.416)
        assert obj.xi_HI == pytest.approx(1.0)
        assert obj.f_HI == pytest.approx(0.83)
        assert obj.col_HI_fact == pytest.approx(1.3e-8)
        assert obj.col_HI == pytest.approx(1.3e-8 * 0.83 * 1.0 / 13.598**2)
        assert obj.temp_HI == pytest.approx(13.598 * c.ev2k)


class TestCosmologyParameters:
    def test_cosmology_parameters(self):
        obj = CosmologyParameters.from_dict({})

        assert not obj.cosmological
        assert obj.h == pytest.approx(0.7)
        assert obj.Omega0 == pytest.approx(0.27)
        assert obj.Omega_B == pytest.approx(0.043)
        assert obj.cmbtemp == pytest.approx(2.726)
        assert obj.zred_0 == pytest.approx(9.0)


class TestPhotoParameters:
    def test_photo_parameters(self):
        obj = PhotoParameters.from_dict({})

        assert obj.sigma_HI_at_ion_freq == pytest.approx(6.30e-18)
        assert obj.minlogtau == pytest.approx(-20)
        assert obj.maxlogtau == 4
        assert obj.num_tau == 20000
        assert not obj.grey
        assert obj.source_type == "blackbody"
        assert not obj.compute_heating_rates
        assert obj.sed_table_path is None


class TestBlackBodySourceParameters:
    def test_black_body_source_parameters(self):
        obj = BlackBodySourceParameters.from_dict({})

        assert obj.Teff == 5e4
        assert obj.cross_section_pl_index == 2.8


class TestSinksParameters:
    def test_sinks_parameters_constant_clumping(self):
        obj = SinksParameters.from_dict({"clumping_model": "constant"})

        assert obj.clumping_model == "constant"
        assert isinstance(obj.clumping, ConstantClumpingParameters)
        assert obj.clumping.value == pytest.approx(5.0)

    def test_sinks_parameters_redshift_clumping(self):
        obj = SinksParameters.from_dict({"clumping_model": "redshift"})

        assert obj.clumping_model == "redshift"
        assert isinstance(obj.clumping, RedshiftClumpingParameters)

    def test_sinks_parameters_density_clumping(self):
        obj = SinksParameters.from_dict({"clumping_model": "density"})

        assert obj.clumping_model == "density"
        assert isinstance(obj.clumping, DensityClumpingParameters)

    def test_sinks_parameters_stochastic_clumping(self):
        obj = SinksParameters.from_dict({"clumping_model": "stochastic"})

        assert obj.clumping_model == "stochastic"
        assert isinstance(obj.clumping, StochasticClumpingParameters)

    def test_sinks_parameters_constant_mfp(self):
        obj = SinksParameters.from_dict({"mfp_model": "constant"})

        assert obj.mfp_model == "constant"
        assert isinstance(obj.mfp, ConstantMfpParameters)
        assert obj.mfp.R_max_cMpc == pytest.approx(15.0)

    def test_sinks_parameters_choudhury09_mfp(self):
        obj = SinksParameters.from_dict({"mfp_model": "Choudhury09"})

        assert obj.mfp_model == "Choudhury09"
        assert isinstance(obj.mfp, Choudhury09MfpParameters)
        assert obj.mfp.A_mfp is None
        assert obj.mfp.eta_mfp is None

    def test_sinks_parameters_worseck14_mfp(self):
        obj = SinksParameters.from_dict({"mfp_model": "Worseck14"})

        assert obj.mfp_model == "Worseck14"
        assert isinstance(obj.mfp, Worseck14MfpParameters)
        assert obj.mfp.A_mfp == pytest.approx(210.0)
        assert obj.mfp.eta_mfp == pytest.approx(-9.0)
        assert obj.mfp.eta1_mfp == pytest.approx(9.0)
        assert obj.mfp.z1_mfp == pytest.approx(6.0)

    def test_sinks_to_yaml_block(self):
        obj = SinksParameters.from_dict({})
        yaml_block = obj.to_yaml_block()

        expected_yaml_block = """
Sinks:
  # Clumping model
  clumping_model: constant
  # Clumping factor for the constant model
  value: 5.0
  # Mean-free-path model
  mfp_model: constant
  # Maximum comoving distance for photons from source
  R_max_cMpc: 15.0
""".strip()

        assert yaml_block == expected_yaml_block


class TestSourcesParameters:
    def test_sources_parameters_fgamma_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_kind": "fgamma"})

        assert obj.fstar_kind == "fgamma"
        assert isinstance(obj.fstar, FgammaSourceParameters)
        assert obj.fstar.fgamma_hm is None
        assert obj.fstar.fgamma_lm is None
        assert obj.ts is None

    def test_sources_parameters_dpl_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_kind": "dpl"})

        assert obj.fstar_kind == "dpl"
        assert isinstance(obj.fstar, DplSourceParameters)
        assert obj.fstar.Nion is None
        assert obj.fstar.f0 is None
        assert obj.fstar.Mt is None
        assert obj.fstar.Mp is None
        assert obj.fstar.g1 is None
        assert obj.fstar.g2 is None
        assert obj.fstar.g3 is None
        assert obj.fstar.g4 is None
        assert obj.ts is None

    def test_sources_parameters_lognorm_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_kind": "lognorm"})

        assert obj.fstar_kind == "lognorm"
        assert isinstance(obj.fstar, LognormSourceParameters)
        assert obj.ts is None

    def test_sources_parameters_muv_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_kind": "Muv"})

        assert obj.fstar_kind == "Muv"
        assert isinstance(obj.fstar, MuvSourceParameters)
        assert obj.fstar.a_s is None
        assert obj.fstar.b_s is None
        assert obj.ts is None

    def test_sources_parameters_constant_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "constant"})

        assert obj.fesc_model == "constant"
        assert isinstance(obj.fesc, ConstantFescParameters)
        assert obj.fesc.f0_esc is None
        assert obj.ts is None

    def test_sources_parameters_power_fest(self):
        obj = SourcesParameters.from_dict({"fesc_model": "power"})

        assert obj.fesc_model == "power"
        assert isinstance(obj.fesc, PowerFescParameters)
        assert obj.fesc.f0_esc is None
        assert obj.fesc.Mp_esc is None
        assert obj.fesc.al_esc is None
        assert obj.ts is None

    def test_sources_parameters_power_obs_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "power_obs"})

        assert obj.fesc_model == "power_obs"
        assert isinstance(obj.fesc, PowerObsFescParameters)
        assert obj.ts is None

    def test_sources_parameters_gelli24_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "Gelli24"})

        assert obj.fesc_model == "Gelli24"
        assert isinstance(obj.fesc, Gelli24FescParameters)
        assert obj.ts is None

    def test_sources_parameters_thesan_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "Thesan"})

        assert obj.fesc_model == "Thesan"
        assert isinstance(obj.fesc, ThesanFescParameters)
        assert obj.ts is None

    def test_sources_parameters_constant_accretion(self):
        obj = SourcesParameters.from_dict({"accretion_model": "constant"})

        assert obj.accretion_model == "constant"
        assert isinstance(obj.accretion, ConstantAccretionParameters)
        assert obj.ts is None

    def test_sources_parameters_exp_accretion_fail(self):
        with pytest.raises(ValueError):
            SourcesParameters.from_dict(
                {"accretion_model": "exp", "bursty_sfr": "unknown"}
            )

    def test_sources_parameters_exp_accretion(self):
        obj = SourcesParameters.from_dict({"accretion_model": "exp"})

        assert obj.accretion_model == "exp"
        assert isinstance(obj.accretion, ExpAccretionParameters)
        assert obj.accretion.alpha_h is None
        assert obj.accretion.bursty_sfr == "no"
        assert obj.accretion.beta1 is None
        assert obj.accretion.beta2 is None
        assert obj.accretion.tB0 is None
        assert obj.accretion.tQ_frac is None
        assert obj.accretion.z0 is None
        assert obj.accretion.t_rnd is None
        assert obj.ts is None

    def test_sources_to_yaml_block(self):
        obj = SourcesParameters.from_dict({})
        yaml_block = obj.to_yaml_block()

        expected_yaml_block = """
Sources:
  # Stellar-to-halo mass relation
  fstar_kind: fgamma
  # Efficiency High-Mass Atomically Cooling Halo (HMACH)
  fgamma_hm: null
  # Efficiency Low-Mass Atomically Cooling Halo (LMACH)
  fgamma_lm: null
  # Escaping photon fraction model
  fesc_model: constant
  # Escape fraction parameter
  f0_esc: null
  # Accretion model
  accretion_model: constant
  # TODO: add description
  ts: null
""".strip()

        assert yaml_block == expected_yaml_block


class TestSimulationParameters:
    def test_from_file(self, tmp_path: Path):
        path = tmp_path / "parameters.yml"

        obj = SimulationParameters()
        obj.to_yaml_file(path)

        new_obj = SimulationParameters.from_file(path)
        assert new_obj == obj

    def test_from_file_partial(self, tmp_path: Path):
        path = tmp_path / "parameters.yml"

        obj = SimulationParameters()
        obj.output.logfile = "custom.log"
        generate_parameter_file(path, sections=(obj.output,))

        # Read from file, but fill in the rest of the parameters with defaults
        new_obj = SimulationParameters.from_file(path)
        assert new_obj == obj
