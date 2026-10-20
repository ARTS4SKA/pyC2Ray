from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest

import pyc2ray.constants as c
from pyc2ray.domain.morton_grouping import MAX_MORTON_BITS
from pyc2ray.parameters import (
    BlackBodySourceParameters,
    ChemistryParameters,
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
    GridParameters,
    InputOutputParameters,
    MaterialParameters,
    MuvSourceParameters,
    PhotoParameters,
    PowerFescParameters,
    PowerLawSourceParameters,
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
    Zackrisson11SourceParameters,
    _ModelRegistry,
)


class TestYmlParameters:
    @dataclass
    class MockParameters(YmlParameters):
        """Mock parameters for testing purposes"""

        SECTION: ClassVar[str] = "Mock"

        # param0 is required
        param0: str = field(metadata={"doc": "A required string parameter"})
        param1: int = field(default=0, metadata={"doc": "An integer parameter"})
        param2: float = field(default=1.0, metadata={"doc": "A float parameter"})
        param3: Literal["a", "b", "c"] = field(
            default="a", metadata={"doc": "A parameter with limited choices [a, b, c]"}
        )
        optparam: str | None = field(
            default=None, metadata={"doc": "An optional parameter"}
        )

    def test_default_no_init(self):
        with pytest.raises(TypeError):
            TestYmlParameters.MockParameters()

    def test_default_missing_required(self):
        with pytest.raises(TypeError):
            TestYmlParameters.MockParameters.from_dict({})

    def test_default(self):
        obj = TestYmlParameters.MockParameters.from_dict({"param0": "required"})

        assert obj.param0 == "required"
        assert obj.param1 == 0
        assert obj.param2 == pytest.approx(1.0)
        assert obj.param3 == "a"
        assert obj.optparam is None

    def test_from_dict_with_warning_default_missing_keys(self, caplog):
        obj = TestYmlParameters.MockParameters.from_dict(
            {
                "param0": "required",
                "param1": 10,
                "param2": 3.14,
                "param5": "ignored",
            },
            warn=True,
        )

        assert obj.param0 == "required"
        assert obj.param1 == 10
        assert obj.param2 == pytest.approx(3.14)
        assert obj.param3 == "a"
        assert obj.optparam is None

        # Warning emitted
        assert not hasattr(obj, "param5")
        assert len(caplog.records) == 2
        assert caplog.records[0].levelno == logging.WARNING

    def test_from_dict_without_warning_default_missing_keys(self, caplog):
        obj = TestYmlParameters.MockParameters.from_dict(
            {
                "param0": "required",
                "param1": 10,
                "param2": 3.14,
                "param5": "ignored",
            },
            warn=False,
        )

        assert obj.param0 == "required"
        assert obj.param1 == 10
        assert obj.param2 == pytest.approx(3.14)
        assert obj.param3 == "a"
        assert obj.optparam is None

        # Warning not emitted
        assert not hasattr(obj, "param5")
        assert len(caplog.records) == 0

    def test_validate_literas(self):
        # Valid value
        for value in ["a", "b", "c"]:
            obj = TestYmlParameters.MockParameters.from_dict(
                {"param0": "required", "param3": value}
            )
            assert obj.param3 == value

        # Invalid value
        with pytest.raises(ValueError):
            TestYmlParameters.MockParameters.from_dict(
                {"param0": "required", "param3": "invalid"}
            )

    def test_to_yaml_block(self):
        obj = TestYmlParameters.MockParameters.from_dict(
            {
                "param0": "required",
                "param1": 10,
                "param2": 3.14,
                "param3": "b",
                "param5": "ignored",
            }
        )
        yaml_block = obj.to_yaml_block()

        expected_yaml_block = """
# ------------------------------------
# Mock parameters for testing purposes
# ------------------------------------
Mock:
  # A required string parameter
  param0: required
  # An integer parameter
  param1: 10
  # A float parameter
  param2: 3.14
  # A parameter with limited choices [a, b, c]
  param3: b
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
        assert obj.param3 == "b"
        assert obj.optparam is None


class ModelParameters(_ModelRegistry["ModelParameters"]):
    """Model parameters"""

    _registry: ClassVar[dict] = {}
    KEY: ClassVar[str] = "model_type"


@dataclass
class ConstantModelParameters(ModelParameters):
    MODEL: ClassVar[str] = "constant"

    value: float = field(default=5.0, metadata={"doc": "A constant parameter"})


class TestModelRegistry:
    @dataclass
    class MockParameters(YmlParameters):
        """Mock parameters for testing purposes"""

        SECTION: ClassVar[str] = "Mock"

        param1: int = field(default=0, metadata={"doc": "An integer parameter"})
        model: ModelParameters = field(default_factory=ModelParameters)

        @classmethod
        def from_dict(cls, yml, warn=True):
            model = ModelParameters.from_model(yml, "constant", warn)
            return super().from_dict({**yml, "model": model}, warn=warn)

    def test_model_registry(self):
        obj = TestModelRegistry.MockParameters.from_dict(
            {"param1": 10, "model": "constant", "value": 1.0}
        )

        assert obj.param1 == 10
        assert isinstance(obj.model, ConstantModelParameters)
        assert obj.model.value == pytest.approx(1.0)

    def test_to_yaml_block(self):
        obj = TestModelRegistry.MockParameters.from_dict(
            {"param1": 10, "model": "constant", "value": 1.0}
        )
        yaml_block = obj.to_yaml_block()

        expected_yaml_block = """
# ------------------------------------
# Mock parameters for testing purposes
# ------------------------------------
Mock:
  # An integer parameter
  param1: 10
  # Model parameters
  model_type: constant
  # A constant parameter
  value: 1.0
""".strip()

        assert yaml_block == expected_yaml_block


class TestInputOutputParameters:
    def test_input_output_parameters(self):
        obj = InputOutputParameters.from_dict({"results_dir": "results"})

        assert str(obj.results_dir) == "results"
        assert obj.log_name == "pyC2Ray.log"
        assert str(obj.log_file) == "results/pyC2Ray.log"
        assert str(obj.inputs_dir) == "."
        assert str(obj.sources_dir) == "."
        assert str(obj.density_dir) == "."

    def test_input_output_parameters_absolute_log_file(self):
        obj = InputOutputParameters.from_dict(
            {"results_dir": "results", "log_name": "other_log.txt"}
        )

        assert str(obj.results_dir) == "results"
        assert obj.log_name == "other_log.txt"
        assert str(obj.log_file) == "results/other_log.txt"
        assert str(obj.inputs_dir) == "."
        assert str(obj.sources_dir) == "."
        assert str(obj.density_dir) == "."

    def test_input_output_parameters_absolute_inputs(self):
        obj = InputOutputParameters.from_dict(
            {
                "results_dir": "results",
                "inputs_dir": "/path/to/inputs",
                "sources_dir": "/path/to/sources",
                "density_dir": "/path/to/density",
            }
        )

        assert str(obj.results_dir) == "results"
        assert str(obj.log_file) == "results/pyC2Ray.log"
        assert str(obj.inputs_dir) == "/path/to/inputs"
        assert str(obj.sources_dir) == "/path/to/sources"
        assert str(obj.density_dir) == "/path/to/density"

    def test_input_output_parameters_relative_inputs(self):
        obj = InputOutputParameters.from_dict(
            {
                "results_dir": "results",
                "inputs_dir": "/path/to/inputs",
                "sources_dir": "sources",
                "density_dir": "density",
            }
        )

        assert str(obj.results_dir) == "results"
        assert str(obj.log_file) == "results/pyC2Ray.log"
        assert str(obj.inputs_dir) == "/path/to/inputs"
        assert str(obj.sources_dir) == "/path/to/inputs/sources"
        assert str(obj.density_dir) == "/path/to/inputs/density"


class TestGridParameters:
    def test_grid_parameters(self):
        obj = GridParameters.from_dict(
            {
                "boxsize": pytest.approx(100.0),
                "meshsize": 256,
                "gpu": True,
                "mpi": False,
                "resume": True,
            }
        )

        assert obj.boxsize == pytest.approx(100.0)
        assert obj.meshsize == 256
        assert obj.gpu
        assert not obj.mpi
        assert obj.resume


class TestRaytracingParameters:
    def test_raytracing_parameters(self):
        obj = RaytracingParameters.from_dict({"source_batch_size": 50})

        assert obj.source_batch_size == 50
        assert obj.loss_fraction == pytest.approx(1e-2)
        assert obj.subboxsize == 128
        assert obj.max_subbox == 1000


class TestDomainDecompositionParameters:
    def test_domain_decomposition_rejects_unknown_grouping_algorithm(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"grouping_algorithm": "unknown"})

    def test_domain_decomposition_rejects_non_positive_max_num_sources_per_group(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"max_num_sources_per_group": -1})

    @pytest.mark.parametrize("value", [2.5, 10.0, True])
    def test_domain_decomposition_rejects_non_integer_max_num_sources_per_group(
        self, value
    ):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict(
                {"max_num_sources_per_group": value}
            )

    @pytest.mark.parametrize("value", [2.5, 10.0, True])
    def test_domain_decomposition_rejects_non_integer_morton_bits(self, value):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"morton_bits": value})

    def test_domain_decomposition_accepts_numpy_integers(self):
        obj = DomainDecompositionParameters.from_dict(
            {"max_num_sources_per_group": np.int64(12), "morton_bits": np.int32(8)}
        )

        assert obj.max_num_sources_per_group == 12
        assert obj.morton_bits == 8

    def test_domain_decomposition_rejects_non_positive_morton_bits(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"morton_bits": -1})

    def test_domain_decomposition_rejects_too_large_morton_bits(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict(
                {"morton_bits": MAX_MORTON_BITS + 1}
            )

    def test_domain_decomposition_accepts_max_morton_bits(self):
        obj = DomainDecompositionParameters.from_dict({"morton_bits": MAX_MORTON_BITS})

        assert obj.morton_bits == MAX_MORTON_BITS

    def test_domain_decomposition_rejects_non_positive_max_memory_cost_per_group(self):
        with pytest.raises(ValueError):
            DomainDecompositionParameters.from_dict({"max_memory_cost_per_group": -1})

    def test_domain_decomposition_parameters_empty(self):
        obj = DomainDecompositionParameters.from_dict({})

        assert not obj.enabled
        assert obj.grouping_algorithm == "morton_incremental"
        assert obj.max_num_sources_per_group == 1000
        assert obj.morton_bits == 10
        assert obj.max_memory_cost_per_group == pytest.approx(50.0e9)


class TestMaterialParameters:
    def test_material_parameters(self):
        obj = MaterialParameters.from_dict({})

        assert obj.temp0 == pytest.approx(1e4)
        assert obj.xHII == pytest.approx(1.0e-5)
        assert obj.avg_dens == pytest.approx(1.87e-7)


class TestChemistryParameters:
    def test_chemistry_parameters(self):
        obj = ChemistryParameters.from_dict({})

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
        assert obj.convergence_fraction == pytest.approx(1e-4)


class TestCosmologyParameters:
    def test_cosmology_parameters(self):
        obj = CosmologyParameters.from_dict({})

        assert not obj.cosmological
        assert obj.h == pytest.approx(0.7)
        assert obj.Omega0 == pytest.approx(0.27)
        assert obj.Omega_B == pytest.approx(0.043)
        assert obj.cmbtemp == pytest.approx(2.726)
        assert obj.zred_0 == pytest.approx(9.0)
        assert obj.abu_h == pytest.approx(0.926)
        assert obj.abu_he == pytest.approx(0.074)
        assert obj.abu_c == pytest.approx(7.1e-7)
        assert obj.mean_molecular == pytest.approx(0.926 + 4.0 * 0.074)


class TestPhotoParameters:
    def test_photo_parameters_blackbody(self):
        obj = PhotoParameters.from_dict({"source_type_model": "blackbody"})

        assert obj.minlogtau == pytest.approx(-20.0)
        assert obj.maxlogtau == pytest.approx(4.0)
        assert obj.num_tau == 2000
        assert not obj.grey
        assert not obj.compute_heating_rates

        assert obj.source_type_model == "blackbody"
        assert isinstance(obj.source_type, BlackBodySourceParameters)
        assert obj.source_type.Teff == pytest.approx(5e4)
        assert obj.source_type.cross_section_pl_index == pytest.approx(2.8)
        assert obj.source_type.sigma_HI_at_ion_freq == pytest.approx(6.30e-18)

    def test_photo_parameters_powerlaw(self):
        obj = PhotoParameters.from_dict({"source_type_model": "powerlaw"})

        assert obj.minlogtau == pytest.approx(-20.0)
        assert obj.maxlogtau == pytest.approx(4.0)
        assert obj.num_tau == 2000
        assert not obj.grey
        assert not obj.compute_heating_rates

        assert obj.source_type_model == "powerlaw"
        assert isinstance(obj.source_type, PowerLawSourceParameters)

    def test_photo_parameters_zackrisson11(self):
        obj = PhotoParameters.from_dict({"source_type_model": "Zackrisson11"})

        assert obj.minlogtau == pytest.approx(-20.0)
        assert obj.maxlogtau == pytest.approx(4.0)
        assert obj.num_tau == 2000
        assert not obj.grey
        assert not obj.compute_heating_rates

        assert obj.source_type_model == "Zackrisson11"
        assert isinstance(obj.source_type, Zackrisson11SourceParameters)
        assert obj.source_type.Teff == pytest.approx(5e4)
        assert obj.source_type.cross_section_pl_index == pytest.approx(2.8)
        assert str(obj.source_type.sed_table_file) == "."


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
        assert obj.mfp.A_mfp == pytest.approx(210.0)
        assert obj.mfp.eta_mfp == pytest.approx(-9.0)

    def test_sinks_parameters_worseck14_mfp(self):
        obj = SinksParameters.from_dict({"mfp_model": "Worseck14"})

        assert obj.mfp_model == "Worseck14"
        assert isinstance(obj.mfp, Worseck14MfpParameters)
        assert obj.mfp.A_mfp == pytest.approx(210.0)
        assert obj.mfp.eta_mfp == pytest.approx(-9.0)
        assert obj.mfp.eta1_mfp == pytest.approx(9.0)
        assert obj.mfp.z1_mfp == pytest.approx(6.0)


class TestSourcesParameters:
    def test_sources_parameters_fgamma_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_model": "fgamma"})

        assert obj.num_ion == 2000
        assert obj.fstar_model == "fgamma"
        assert isinstance(obj.fstar, FgammaSourceParameters)
        assert obj.fstar.fgamma_hm == pytest.approx(30.0)
        assert obj.fstar.fgamma_lm == pytest.approx(0.0)

    def test_sources_parameters_dpl_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_model": "dpl"})

        assert obj.num_ion == 2000
        assert obj.fstar_model == "dpl"
        assert isinstance(obj.fstar, DplSourceParameters)
        assert obj.fstar.f0 == pytest.approx(0.02)
        assert obj.fstar.Mt == pytest.approx(1e10)
        assert obj.fstar.Mp == pytest.approx(1e10)
        assert obj.fstar.g1 == pytest.approx(-0.3)
        assert obj.fstar.g2 == pytest.approx(-0.3)
        assert obj.fstar.g3 == pytest.approx(0.0)
        assert obj.fstar.g4 == pytest.approx(-0.0)

    def test_sources_parameters_muv_source_fstar(self):
        obj = SourcesParameters.from_dict({"fstar_model": "Muv"})

        assert obj.num_ion == 2000
        assert obj.fstar_model == "Muv"
        assert isinstance(obj.fstar, MuvSourceParameters)
        assert obj.fstar.f0 == pytest.approx(0.02)
        assert obj.fstar.Mt == pytest.approx(1e10)
        assert obj.fstar.Mp == pytest.approx(1e10)
        assert obj.fstar.g1 == pytest.approx(-0.3)
        assert obj.fstar.g2 == pytest.approx(-0.3)
        assert obj.fstar.g3 == pytest.approx(0.0)
        assert obj.fstar.g4 == pytest.approx(-0.0)
        assert obj.fstar.a_s == pytest.approx(-1.0 / 3.0)
        assert obj.fstar.b_s == pytest.approx(9.0 / 2.0)

    def test_sources_parameters_constant_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "constant"})

        assert obj.num_ion == 2000
        assert obj.fesc_model == "constant"
        assert isinstance(obj.fesc, ConstantFescParameters)
        assert obj.fesc.f0_esc == pytest.approx(0.1)

    def test_sources_parameters_power_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "power"})

        assert obj.num_ion == 2000
        assert obj.fesc_model == "power"
        assert isinstance(obj.fesc, PowerFescParameters)
        assert obj.fesc.f0_esc == pytest.approx(0.1)
        assert obj.fesc.Mp_esc == pytest.approx(1e10)
        assert obj.fesc.al_esc == pytest.approx(-0.3)

    def test_sources_parameters_power_obs_fesc(self):
        obj = SourcesParameters.from_dict({"fesc_model": "power_obs"})

        assert obj.num_ion == 2000
        assert obj.fesc_model == "power_obs"
        assert isinstance(obj.fesc, PowerObsFescParameters)
        assert obj.fesc.f0_esc == pytest.approx(0.1)
        assert obj.fesc.Mp_esc == pytest.approx(1e10)
        assert obj.fesc.al_esc == pytest.approx(-0.3)
        assert obj.fesc.al_esc_z == pytest.approx(1.0)
        assert obj.fesc.zp_esc == pytest.approx(0.0)

    def test_sources_parameters_thesan_fesc(self):
        obj = SourcesParameters.from_dict(
            {"fesc_model": "thesan", "tables_dir": "/path/to/thesan/"}
        )

        assert obj.num_ion == 2000
        assert obj.fesc_model == "thesan"
        assert isinstance(obj.fesc, ThesanFescParameters)
        assert str(obj.fesc.tables_dir) == "/path/to/thesan"

    def test_sources_parameters_constant_accretion(self):
        obj = SourcesParameters.from_dict({"accretion_model": "constant"})

        assert obj.num_ion == 2000
        assert obj.accretion_model == "constant"
        assert isinstance(obj.accretion, ConstantAccretionParameters)

    def test_sources_parameters_exp_accretion_fail(self):
        with pytest.raises(ValueError):
            SourcesParameters.from_dict(
                {"accretion_model": "exp", "bursty_sfr": "unknown"}
            )

    def test_sources_parameters_exp_accretion(self):
        obj = SourcesParameters.from_dict({"accretion_model": "exp"})

        assert obj.accretion_model == "exp"
        assert isinstance(obj.accretion, ExpAccretionParameters)
        assert obj.accretion.bursty_sfr == "no"
        assert obj.accretion.alpha_h == pytest.approx(0.79)
        assert obj.accretion.beta1 == pytest.approx(0.1)
        assert obj.accretion.beta2 == pytest.approx(1.5)
        assert obj.accretion.tB0 == pytest.approx(200.0)
        assert obj.accretion.tQ_frac == pytest.approx(1.6)
        assert obj.accretion.z0 == pytest.approx(30.0)
        assert obj.accretion.t_rnd == pytest.approx(0)


class TestSimulationParameters:
    @pytest.fixture(scope="class")
    def sim_params(self) -> SimulationParameters:
        return SimulationParameters(
            InputOutputParameters.from_dict({"results_dir": "results"}),
            GridParameters.from_dict(
                {
                    "boxsize": 100.0,
                    "meshsize": 256,
                    "gpu": True,
                    "mpi": False,
                    "resume": True,
                }
            ),
            RaytracingParameters.from_dict({"source_batch_size": 50}),
            DomainDecompositionParameters.from_dict({}),
            MaterialParameters.from_dict({}),
            ChemistryParameters.from_dict({}),
            CosmologyParameters.from_dict({}),
            PhotoParameters.from_dict({}),
            SinksParameters.from_dict({}),
            SourcesParameters.from_dict({}),
        )

    def test_from_file(self, tmp_path: Path, sim_params: SimulationParameters):
        path = tmp_path / "parameters.yml"

        sim_params.to_yaml_file(path)

        new_obj = SimulationParameters.from_file(path)
        assert new_obj == sim_params
