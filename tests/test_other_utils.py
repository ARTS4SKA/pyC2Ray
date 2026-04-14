import pytest

from pyc2ray.utils.other_utils import display_seconds, distribute_jobs


def test_display_seconds():
    assert display_seconds(0) == "0:00:00"
    assert display_seconds(1) == "0:00:01"
    assert display_seconds(60) == "0:01:00"
    assert display_seconds(3600) == "1:00:00"
    assert display_seconds(3661) == "1:01:01"


@pytest.mark.parametrize("jobs", [10, 100, 1000])
@pytest.mark.parametrize("procs", [7, 13, 29])
def test_distribute_jobs(jobs: int, procs: int):
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
