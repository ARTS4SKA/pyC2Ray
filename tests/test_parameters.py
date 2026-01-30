from dataclasses import dataclass, fields
from pathlib import Path

import pytest

import pyc2ray.constants as c
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


@pytest.fixture(scope="session")
def params_file(data_dir: Path) -> Path:
    return data_dir / "parameters.yml"


@dataclass
class MockParameters(YmlParameters):
    param1: int
    param2: float
    param3: str = "default"


class TestYmlParameters:
    def test_mock_parameters(self):
        obj = MockParameters.from_dict(
            {"param1": 10, "param2": 3.14, "param3": "custom", "param4": "ignored"}
        )

        assert obj.param1 == 10
        assert obj.param2 == 3.14
        assert obj.param3 == "custom"
        assert not hasattr(obj, "param4")

    def test_output_parameters(self, params_file: Path):
        obj = OutputParameters.from_file(params_file, "Output")

        assert obj.results_basename == "/home/username/data"
        assert obj.logfile == "pyC2Ray.log"
        assert obj.inputs_basename is None
        assert obj.sources_basename is None
        assert obj.density_basename is None

    def test_grid_parameters(self, params_file: Path):
        obj = GridParameters.from_file(params_file, "Grid")

        assert obj.boxsize == 1.62022035
        assert obj.meshsize == 256
        assert obj.gpu == 0
        assert obj.mpi == 0
        assert obj.resume == 0

    def test_raytracing_parameters(self, params_file: Path):
        obj = RaytracingParameters.from_file(params_file, "Raytracing")

        assert obj.loss_fraction == 1e-2
        assert obj.subboxsize == 128
        assert obj.max_subbox == 1000
        assert obj.source_batch_size == 1
        assert obj.convergence_fraction == 1e-4

    def test_material_parameters(self, params_file: Path):
        obj = MaterialParameters.from_file(params_file, "Material")

        assert obj.temp0 == 1e4
        assert obj.xh0 == 1.2e-3
        assert obj.avg_dens == 1.87e-7

    def test_cgs_parameters(self, params_file: Path):
        obj = CGSParameters.from_file(params_file, "CGS")

        assert obj.albpow == -0.7
        assert obj.bh00 == 2.59e-13
        assert obj.alcpow == -0.672
        assert obj.eth0 == 13.598
        assert obj.ethe0 == 24.587
        assert obj.ethe1 == 54.416
        assert obj.xih0 == 1.0
        assert obj.fh0 == 0.83
        assert obj.colh0_fact == 1.3e-8
        assert obj.colh0 == 1.3e-8 * 0.83 * 1.0 / 13.598**2
        assert obj.temph0 == 13.598 * c.ev2k

    def test_cosmology_parameters(self, params_file: Path):
        obj = CosmologyParameters.from_file(params_file, "Cosmology")

        assert obj.cosmological == 0
        assert obj.h == 1.0
        assert obj.Omega0 == 0.27
        assert obj.Omega_B == 0.044
        assert obj.cmbtemp == 2.726
        assert obj.zred_0 == 9.0

    def test_abundance_parameters(self, params_file: Path):
        obj = AbundancesParameters.from_file(params_file, "Abundances")

        assert obj.abu_h == 0.926
        assert obj.abu_he == 0.074
        assert obj.abu_c == 7.1e-7
        assert obj.mean_molecular == 0.926 + 4.0 * 0.074

    def test_photo_parameters(self, params_file: Path):
        obj = PhotoParameters.from_file(params_file, "Photo")

        assert obj.sigma_HI_at_ion_freq == 6.30e-18
        assert obj.minlogtau == -20
        assert obj.maxlogtau == 4
        assert obj.NumTau == 20000
        assert obj.grey == 1
        assert obj.SourceType == "blackbody"
        assert obj.compute_heating_rates == 0

    def test_sinks_parameters(self, params_file: Path):
        obj = SinksParameters.from_file(params_file, "Sinks")

        assert obj.clumping_model == "constant"
        assert obj.clumping == 5.0
        assert obj.mfp_model == "constant"
        assert obj.R_max_cMpc == 15.0
        assert obj.A_mfp == 175.0
        assert obj.eta_mfp == -4.4
        assert obj.z1_mfp is None
        assert obj.eta1_mfp is None

    def test_black_body_parameters(self, params_file: Path):
        obj = BlackBodyParameters.from_file(params_file, "BlackBodySource")

        assert obj.Teff == 5e4
        assert obj.cross_section_pl_index == 2.8

    def test_sources_parameters_empty(self):
        obj = SourcesParameters.from_dict({})

        assert obj.fstar_kind == "fgamma"
        assert obj.fesc_model == "constant"
        assert obj.accretion_model == "constant"
        assert obj.bursty_sfr == "no"

        for field in fields(obj):
            if field.name not in (
                "fstar_kind",
                "fesc_model",
                "accretion_model",
                "bursty_sfr",
            ):
                assert getattr(obj, field.name) is None

    def test_sources_parameters(self, params_file: Path):
        obj = SourcesParameters.from_file(params_file, "Sources")

        assert obj.fstar_kind == "dpl"
        assert obj.fgamma_hm == 30
        assert obj.fgamma_lm == 0.0
        assert obj.Nion == 2000
        assert obj.f0 == 0.1
        assert obj.Mt == 1e10
        assert obj.Mp == 1e10
        assert obj.g1 == -0.3
        assert obj.g2 == -0.3
        assert obj.g3 == 0.0
        assert obj.g4 == 0.0
        assert obj.a_s is None
        assert obj.b_s is None
        assert obj.fesc_model == "constant"
        assert obj.f0_esc == 0.02
        assert obj.Mp_esc == 1e10
        assert obj.al_esc == -0.25
        assert obj.accretion_model == "EXP"
        assert obj.alpha_h == 0.79
        assert obj.bursty_sfr == "no"
        assert obj.beta1 == 0.1
        assert obj.beta2 == 1.5
        assert obj.tB0 == 200.0
        assert obj.tQ_frac == 1.6
        assert obj.z0 == 30.0
        assert obj.t_rnd == 0
