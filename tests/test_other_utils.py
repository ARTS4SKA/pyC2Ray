import numpy as np
import pytest

from pyc2ray.utils.other_utils import (
    display_seconds,
    distribute_jobs,
    find_redshit_index,
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


@pytest.fixture(scope="module")
def zreds() -> np.ndarray:
    return np.round(np.geomspace(20, 1, num=10), decimals=3)


# TODO: Skip 10.280 until we agree on behavior
@pytest.mark.parametrize("z", [30.0, 20.0, 10.278, 10.276, 10.0, 5.0, 1.0, 0.0])
def test_find_redshit_index(zreds: np.ndarray, z: float) -> None:
    idx = int(np.argmin(np.abs(zreds - z)))
    assert find_redshit_index(zreds, z) == idx
