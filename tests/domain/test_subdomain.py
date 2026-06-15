"""Unit tests focused on Subdomain._assign_groups_to_ranks."""

from __future__ import annotations

import numpy as np

from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain


class DummyComm:
    """Small MPI-like communicator stub for assignment tests."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def scatter(self, data: object, root: int = 0) -> object:
        # _assign_groups_to_ranks tests never call scatter, but the method is
        # required to satisfy the Subdomain communicator protocol for typing.
        return data


def _source(sid: int) -> Source:
    return Source(
        id=sid,
        pos=np.array([0.0, 0.0, 0.0], dtype=float),
        strength=1.0,
        radius=1.0,
    )


def _group(gid: int, comp_cost: float) -> SourceGroup:
    src = _source(gid)
    return SourceGroup(
        id=gid,
        sources=[src],
        center=np.array([0.0, 0.0, 0.0], dtype=float),
        radius=1.0,
        bbox_min=np.array([-1.0, -1.0, -1.0], dtype=float),
        bbox_max=np.array([1.0, 1.0, 1.0], dtype=float),
        mem_cost=comp_cost,
        comp_cost=comp_cost,
    )


def test_assign_groups_to_ranks_balances_by_comp_cost() -> None:
    subdomain = Subdomain(comm=DummyComm(rank=0, size=2))
    groups = [
        _group(0, 9.0),
        _group(1, 8.0),
        _group(2, 7.0),
        _group(3, 6.0),
        _group(4, 5.0),
        _group(5, 4.0),
        _group(6, 3.0),
    ]

    rank_groups, rank_costs = subdomain._assign_groups_to_ranks(groups)

    assert len(rank_groups) == 2
    assert rank_costs == [20.0, 22.0]
    assert [g.id for g in rank_groups[0]] == [0, 3, 4]
    assert [g.id for g in rank_groups[1]] == [1, 2, 5, 6]


def test_assign_groups_to_ranks_handles_empty_groups() -> None:
    subdomain = Subdomain(comm=DummyComm(rank=0, size=3))

    rank_groups, rank_costs = subdomain._assign_groups_to_ranks([])

    assert rank_groups == [[], [], []]
    assert rank_costs == [0.0, 0.0, 0.0]
