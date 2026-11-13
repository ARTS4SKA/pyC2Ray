from pathlib import Path

import numpy as np
import pytest

from pyc2ray.c2ray_base import C2Ray
from pyc2ray.parameters import (
    ChemistryParameters,
    CosmologyParameters,
    DomainDecompositionParameters,
    GridParameters,
    InputOutputParameters,
    MaterialParameters,
    PhotoParameters,
    RaytracingParameters,
    SimulationParameters,
    SinksParameters,
    SourcesParameters,
)

MESHSIZE = 8


class C2RayResume(C2Ray):
    """Minimal C2Ray that skips the setup not involved in resuming"""

    def _sources_init(self) -> None:
        pass

    def _radiation_init(self) -> None:
        pass


@pytest.fixture
def make_paramfile(tmp_path: Path, zred_array: np.ndarray):
    """Write a parameter file with results/inputs in tmp_path and the given resume flag"""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    np.savetxt(
        inputs_dir / "redshift_checkpoints.txt",
        np.column_stack([np.arange(len(zred_array)), zred_array]),
    )

    def _make(results_dir: Path, resume: bool) -> Path:
        paramfile = tmp_path / "parameters.yml"
        SimulationParameters(
            InputOutputParameters.from_dict(
                {"results_dir": results_dir, "inputs_dir": inputs_dir}
            ),
            GridParameters.from_dict(
                {
                    "boxsize": 100.0,
                    "meshsize": MESHSIZE,
                    "gpu": True,
                    "mpi": False,
                    "resume": resume,
                }
            ),
            RaytracingParameters.from_dict({"source_batch_size": 10}),
            DomainDecompositionParameters.from_dict(),
            MaterialParameters.from_dict(),
            ChemistryParameters.from_dict(),
            CosmologyParameters.from_dict(),
            PhotoParameters.from_dict(),
            SinksParameters.from_dict(),
            SourcesParameters.from_dict(),
        ).to_yaml_file(paramfile)
        return paramfile

    return _make


def write_outputs(results_dir: Path, zreds: np.ndarray) -> dict[float, np.ndarray]:
    """Write fake xfrac/IonRates outputs and return the xfrac arrays by redshift"""
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    xfracs = {}
    for z in zreds:
        xh = rng.random((MESHSIZE,) * 3)
        np.save(results_dir / f"xfrac_z{z:.3f}.npy", xh)
        np.save(results_dir / f"IonRates_z{z:.3f}.npy", 2 * xh)
        xfracs[z] = xh
    return xfracs


def test_no_resume(make_paramfile, tmp_path: Path, zred_array: np.ndarray) -> None:
    results_dir = tmp_path / "results"
    write_outputs(results_dir, zred_array[:3])
    sim = C2RayResume(make_paramfile(results_dir, resume=False))

    assert not sim.resume
    assert sim.zred == zred_array[0]
    assert np.array_equal(sim.idx_zred, np.arange(len(zred_array)))
    assert np.array_equal(sim.zred_array, zred_array)
    assert np.all(sim.xh == sim.parameters.material.xHII)
    assert np.all(sim.phi_ion == 0)


def test_resume_from_last_output(
    make_paramfile, tmp_path: Path, zred_array: np.ndarray
) -> None:
    results_dir = tmp_path / "results"
    xfracs = write_outputs(results_dir, zred_array[:3])
    sim = C2RayResume(make_paramfile(results_dir, resume=True))

    z_last = zred_array[2]
    assert sim.resume
    assert sim.zred == z_last
    assert np.array_equal(sim.idx_zred, np.arange(2, len(zred_array)))
    assert np.array_equal(sim.zred_array, zred_array[2:])
    assert np.isclose(sim.time, sim.zred2time(z_last))
    assert np.array_equal(sim.xh, xfracs[z_last])
    assert np.array_equal(sim.phi_ion, 2 * xfracs[z_last])


def test_resume_empty_output_starts_fresh(
    make_paramfile, tmp_path: Path, zred_array: np.ndarray
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sim = C2RayResume(make_paramfile(results_dir, resume=True))

    assert not sim.resume
    assert sim.zred == zred_array[0]
    assert np.array_equal(sim.zred_array, zred_array)
    assert np.all(sim.xh == sim.parameters.material.xHII)


def test_dilution_factor(
    make_paramfile, tmp_path: Path, zred_array: np.ndarray
) -> None:
    sim = C2RayResume(make_paramfile(tmp_path / "results", resume=False))

    assert sim.dilution_factor(sim.zred) == 1.0
    expected = (1 + zred_array) / (1 + sim.zred)
    assert np.allclose([sim.dilution_factor(z) for z in zred_array], expected)
