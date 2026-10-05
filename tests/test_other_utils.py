from pathlib import Path

import numpy as np
import pytest

from pyc2ray.utils.other_utils import (
    display_seconds,
    distribute_jobs,
    find_closest_redshift_index,
    get_extension_in_folder,
    get_redshift_list_from_output,
    get_resume_redshift_from_output,
)


def test_display_seconds() -> None:
    assert display_seconds(0) == "0:00:00"
    assert display_seconds(1) == "0:00:01"
    assert display_seconds(60) == "0:01:00"
    assert display_seconds(3600) == "1:00:00"
    assert display_seconds(3661) == "1:01:01"


@pytest.mark.parametrize("jobs", [10, 100, 1000])
@pytest.mark.parametrize("procs", [7, 13, 29])
def test_distribute_jobs(jobs: int, procs: int) -> None:
    tot = 0
    prev = 0
    expected = jobs // procs
    for rank in range(procs):
        chunk = distribute_jobs(jobs, procs, rank)
        assert chunk.step is None or chunk.step == 1
        assert prev == chunk.start

        nitems = chunk.stop - chunk.start
        assert nitems in (expected, expected + 1)

        prev = chunk.stop
        tot += nitems

    assert prev == jobs
    assert tot == jobs


def test_get_extension_in_folder(tmp_path: Path) -> None:
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    (tmp_path / "foo1.csv").touch()
    (tmp_path / "foo2.csv").touch()

    with pytest.raises(ValueError):
        get_extension_in_folder(tmp_path, "invalid")

    assert get_extension_in_folder(tmp_path, "file") == ".txt"
    assert get_extension_in_folder(tmp_path, "foo") == ".csv"


@pytest.fixture(scope="module")
def zred_array() -> np.ndarray:
    return np.geomspace(20, 1, num=10)


def touch_z_files(
    tmp_path: Path, zred_array: np.ndarray, prefix: str, suffix: str
) -> Path:
    for z in zred_array:
        (tmp_path / f"{prefix}z{z:.3f}{suffix}").touch()
    return tmp_path


def test_get_redshift_list_from_output_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        get_redshift_list_from_output(tmp_path, "xfrac_")


def test_get_redshift_list_from_output_npy(
    tmp_path: Path, zred_array: np.ndarray
) -> None:
    xfrac_files = touch_z_files(tmp_path, zred_array, "xfrac_", ".npy")
    result = get_redshift_list_from_output(xfrac_files, "xfrac_")
    assert np.allclose(np.round(zred_array, 3), result)


def test_get_resume_redshift_from_output_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        get_resume_redshift_from_output(tmp_path, "xfrac_")


def test_get_resume_redshift_from_output_npy(
    tmp_path: Path, zred_array: np.ndarray
) -> None:
    xfrac_files = touch_z_files(tmp_path, zred_array[0:7], "xfrac_", ".npy")
    resume = get_resume_redshift_from_output(xfrac_files, "xfrac_")
    assert resume == 2.714


@pytest.mark.parametrize(
    "idx,z",
    enumerate(
        [
            20.000,
            14.337,
            10.278,
            7.368,
            5.282,
            3.786,
            2.714,
            1.946,
            1.395,
            1.0,
        ]
    ),
)
def test_find_closest_redshift_index(
    zred_array: np.ndarray,
    idx: int,
    z: float,
) -> None:
    assert find_closest_redshift_index(zred_array, z) == idx
