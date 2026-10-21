from pathlib import Path

import numpy as np
import pytest
import yaml

from pyc2ray.c2ray_base import C2Ray

MESHSIZE = 8


class C2RayResume(C2Ray):
    """Minimal C2Ray that skips the setup not involved in resuming"""

    def _sources_init(self) -> None:
        pass

    def _radiation_init(self) -> None:
        pass

    def _sinks_init(self) -> None:
        pass


@pytest.fixture
def make_paramfile(data_dir: Path, tmp_path: Path, zred_array: np.ndarray):
    """Write a parameter file with results/inputs in tmp_path and the given resume flag"""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    np.savetxt(
        inputs_dir / "redshift_checkpoints.txt",
        np.column_stack([np.arange(len(zred_array)), zred_array]),
    )

    def _make(results_dir: Path, resume: bool) -> Path:
        params = yaml.safe_load((data_dir / "parameters.yml").read_text())
        params["Output"]["results_basename"] = str(results_dir)
        params["Output"]["inputs_basename"] = str(inputs_dir)
        params["Grid"]["meshsize"] = MESHSIZE
        params["Grid"]["resume"] = int(resume)
        params["DomainDecomposition"]["enabled"] = 0
        paramfile = tmp_path / "parameters.yml"
        paramfile.write_text(yaml.safe_dump(params))
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
    assert np.array_equal(sim.zred_array, zred_array)
    assert np.all(sim.xh == sim.material_params.xh0)
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
    assert np.all(sim.xh == sim.material_params.xh0)


def test_resume_missing_output_dir_raises(make_paramfile, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        C2RayResume(make_paramfile(tmp_path / "typo", resume=True))


def test_dilution_factor(
    make_paramfile, tmp_path: Path, zred_array: np.ndarray
) -> None:
    sim = C2RayResume(make_paramfile(tmp_path / "results", resume=False))

    assert sim.dilution_factor(sim.zred) == 1.0
    expected = (1 + zred_array) / (1 + sim.zred)
    assert np.allclose([sim.dilution_factor(z) for z in zred_array], expected)
