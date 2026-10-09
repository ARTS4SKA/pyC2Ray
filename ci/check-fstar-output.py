import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pyc2ray.utils.other_utils import find_redshit_index

TEXT_FILES = "parameters.yml", "pyC2Ray.log", "PhotonCounts2.txt"


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
        start_idx = find_redshit_index(zred_array, z_start)

    end_idx = len(zred_array)
    if z_end is not None:
        end_idx = find_redshit_index(zred_array, z_end)

    # First redshift is the initial condition and the last redshift is not saved
    return zred_array[start_idx + 1 : end_idx - 1]


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

    phions: set[str] = set()
    xfracs: set[str] = set()

    errors: list[str] = []
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


def main(outdir: Path, z_start: float | None, z_end: float | None) -> int:
    print(f"CHECKING: Simulation output in {outdir}")
    if not outdir.is_dir():
        print(f"FAIL: {outdir} is not a directory", file=sys.stderr)
        return 1

    errors: list[str] = []

    # Check presence of text files
    errors.extend(check_text_files(outdir))

    # Check the correctness of the data files
    errors.extend(check_data_files(outdir, z_start, z_end))

    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    if not errors:
        print("OK: Everything looks good")

    return 1 if errors else 0


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Check the output directory of the F* CI simulation."
    )

    parser.add_argument("outdir", type=Path)
    parser.add_argument("--z-start", type=float, help="Starting redshift (optional)")
    parser.add_argument("--z-end", type=float, help="Ending redshift (optional)")

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**args))
