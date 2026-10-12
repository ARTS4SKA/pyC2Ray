import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pyc2ray.utils.other_utils import find_closest_redshift_index


def check_text_files(outdir: Path) -> list[str]:
    errors: list[str] = []
    for name in ("parameters.yml", "pyC2Ray.log", "PhotonCounts2.txt"):
        path = outdir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{name} missing or empty")
    return errors


def expected_redshifts(
    checkpoints: Path, z_start: float | None, z_end: float | None
) -> np.ndarray:
    _, zred_array = np.loadtxt(checkpoints, unpack=True)

    start_idx = 0
    if z_start is not None:
        start_idx = find_closest_redshift_index(zred_array, z_start)

    end_idx = len(zred_array)
    if z_end is not None:
        end_idx = find_closest_redshift_index(zred_array, z_end)

    # First redshift is the initial condition and the last redshift is not saved
    return zred_array[start_idx : end_idx + 1]


def check_data_files(
    outdir: Path, z_start: float | None, z_end: float | None
) -> list[str]:
    with open(outdir / "parameters.yml", "r") as f:
        data = yaml.safe_load(f)

    # .npy has a 128 byte header before data (float64)
    mesh_size = data["Grid"]["meshsize"]
    expected_size = 128 + mesh_size**3 * 8

    checkpoints = Path(data["Output"]["inputs_basename"], "redshift_checkpoints.txt")
    zreds = expected_redshifts(checkpoints, z_start, z_end)

    errors: list[str] = []
    with open(outdir / "PhotonCounts2.txt", "rb") as f:
        if (counts := len(list(f))) != len(zreds) + 1:
            errors.append(
                f"PhotonCounts2.txt has {counts} lines, expected {len(zreds) + 1}"
            )

    phions: set[str] = set()
    xfracs: set[str] = set()
    for z in zreds:
        zstr = f"{z:.3f}"
        for fset, prefix in [(phions, "IonRates"), (xfracs, "xfrac")]:
            path = outdir / f"{prefix}_z{zstr}.npy"
            if not path.is_file():
                errors.append(f"{prefix} file missing z={zstr}")
            elif (size := path.stat().st_size) != expected_size:
                errors.append(
                    f"{path} file has unexpected size {size} B for mesh size {mesh_size}"
                )
            else:
                fset.add(zstr)

    if only_ion := sorted(phions - xfracs):
        errors.append(f"IonRates without xfrac: {only_ion}")

    if only_xfrac := sorted(xfracs - phions):
        errors.append(f"xfrac without IonRates: {only_xfrac}")

    return errors


def check_ionization_history(results_dir: Path, reference_dir: Path) -> list[str]:
    errors: list[str] = []

    summary: dict[float, tuple[float, float]] = {}
    for xfrac in results_dir.glob("xfrac_z*.npy"):
        z = float(xfrac.stem.split("_z")[-1])
        xfrac_ref = results_dir / xfrac.name
        if not xfrac_ref.is_file():
            errors.append(f"Fraction file for {z=:.3f} missing in reference directory")
            continue

        xh_ref = np.load(xfrac_ref)
        xh = np.load(xfrac)
        summary[z] = (xh_ref.mean(), xh.mean())
        if not np.allclose(xh, xh_ref, rtol=1e-6, atol=1e-8):
            errors.append(f"Fraction {z=:.3f} does not match reference")

    # Print ionization history summary
    for z in sorted(summary, reverse=True):
        xh_ref_mean, xh_mean = summary[z]
        print(f"z={z:.3f}: mean fraction {xh_mean:.8e} vs reference {xh_ref_mean:.8e}")

    return errors


def main(
    results_dir: Path, reference_dir: Path, z_start: float | None, z_end: float | None
) -> int:
    print(f"CHECKING: Simulation output in {results_dir}")
    if not results_dir.is_dir():
        print(f"FAIL: Results directory '{results_dir}' not found", file=sys.stderr)
        return 1
    if not reference_dir.is_dir():
        print(f"FAIL: Reference directory '{reference_dir}' not found", file=sys.stderr)
        return 1

    errors: list[str] = []

    # Check presence of text files
    errors.extend(check_text_files(results_dir))

    # Check the correctness of the data files
    errors.extend(check_data_files(results_dir, z_start, z_end))

    # Check the physics of the simulation
    errors.extend(check_ionization_history(results_dir, reference_dir))

    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    if not errors:
        print("OK: Everything looks good")

    return 1 if errors else 0


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Check the output directory of the F* CI simulation."
    )

    parser.add_argument("results_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--z-start", type=float, help="Starting redshift (optional)")
    parser.add_argument("--z-end", type=float, help="Ending redshift (optional)")

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**args))
