"""Unit tests for DomainDecompositionHandler group assignment and update_decomposition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyc2ray.domain.domain_decomposition_handler import DomainDecompositionHandler
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.parameters import DomainDecompositionParameters


class DummyComm:
    """Small MPI-like communicator stub for the handler tests."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def scatter(self, data: Sequence[Any] | None, root: int = 0) -> Any:
        # Mimic MPI scatter: each rank receives its own slot of the per-rank list.
        if data is None:
            return None
        return data[self._rank]


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
    handler = DomainDecompositionHandler(comm=DummyComm(rank=0, size=2))
    groups = [
        _group(0, 9.0),
        _group(1, 8.0),
        _group(2, 7.0),
        _group(3, 6.0),
        _group(4, 5.0),
        _group(5, 4.0),
        _group(6, 3.0),
    ]

    rank_groups, rank_costs = handler._assign_groups_to_ranks(groups)

    assert len(rank_groups) == 2
    assert rank_costs == [20.0, 22.0]
    assert [g.id for g in rank_groups[0]] == [0, 3, 4]
    assert [g.id for g in rank_groups[1]] == [1, 2, 5, 6]


def test_assign_groups_to_ranks_handles_empty_groups() -> None:
    handler = DomainDecompositionHandler(comm=DummyComm(rank=0, size=3))

    rank_groups, rank_costs = handler._assign_groups_to_ranks([])

    assert rank_groups == [[], [], []]
    assert rank_costs == [0.0, 0.0, 0.0]


def _dd_params(max_num_sources_per_group: int = 4) -> DomainDecompositionParameters:
    return DomainDecompositionParameters(
        enabled=True,
        grouping_algorithm="morton",
        max_num_sources_per_group=max_num_sources_per_group,
        morton_bits=10,
    )


def test_update_decomposition_rebuilds_if_inputs_change() -> None:
    """update_decomposition must (re)build exactly when the grouping inputs change.

    A rebuild reassigns the subdomains list; a reuse leaves it untouched. We use that
    identity to distinguish the two, since update_decomposition returns nothing.
    """
    handler = DomainDecompositionHandler(comm=DummyComm(rank=0, size=1))
    params = _dd_params()

    def update(
        src_pos: np.ndarray, src_flux: np.ndarray, r_max: float = 2.0, dr: float = 1.0
    ) -> None:
        handler.update_decomposition(
            cell_size=dr,
            src_pos=src_pos,
            src_flux=src_flux,
            N=16,
            R_max_LLS=r_max,
            src_batch_size=8,
            num_tau=100,
            is_domain_periodic=True,
            domain_decomposition_params=params,
        )

    src_pos = np.array([[4, 4, 4], [6, 6, 6]], dtype=np.int32)
    src_flux = np.array([1.0, 2.0], dtype=np.float64)

    # First call builds the decomposition.
    update(src_pos, src_flux)
    built = handler.subdomains

    # Identical inputs: the cached decomposition is reused (subdomains untouched).
    update(src_pos, src_flux)
    assert handler.subdomains is built

    # Changed source configuration triggers a rebuild (subdomains reassigned).
    moved_pos = np.array([[4, 4, 4], [10, 10, 10]], dtype=np.int32)
    update(moved_pos, src_flux)
    assert handler.subdomains is not built
    rebuilt = handler.subdomains

    # A cell-size-only change must NOT rebuild: the grouping is invariant under scaling.
    update(moved_pos, src_flux, dr=5.0)
    assert handler.subdomains is rebuilt


def test_every_source_survives_the_decomposition() -> None:
    """No source may be lost or duplicated when sources are split into groups.

    Four well-separated clusters of five sources each, with a group cap of two, so the
    decomposition splits on both criteria: the enclosing spheres of different clusters do
    not intersect, and within a cluster the cap closes the group early. On a single rank
    every group is assigned locally, so the subdomains of this rank must between them hold
    each source exactly once, with its position intact.
    """
    handler = DomainDecompositionHandler(comm=DummyComm(rank=0, size=1))

    cell_size = 1.0
    # Radius 0.6 cells: neighbouring cells along x intersect (centres 1.0 apart, radii sum
    # to 1.2) while the clusters, 4 cells apart in y and z, do not.
    r_max_lls = 0.6
    src_pos = np.array(
        [[x, plane, plane] for plane in (1, 5, 9, 13) for x in (1, 2, 3, 4, 5)],
        dtype=np.int32,
    )
    src_flux = np.arange(1.0, len(src_pos) + 1.0, dtype=np.float64)

    handler.update_decomposition(
        cell_size=cell_size,
        src_pos=src_pos,
        src_flux=src_flux,
        N=16,
        R_max_LLS=r_max_lls,
        src_batch_size=8,
        num_tau=100,
        is_domain_periodic=True,
        domain_decomposition_params=_dd_params(max_num_sources_per_group=2),
    )

    subdomains = handler.get_subdomains()
    grouped_sources = [s for sd in subdomains for s in sd.source_group.sources]

    # Guard against a vacuous pass: the cap must actually have split the sources.
    assert len(subdomains) > 1
    assert all(len(sd.source_group.sources) <= 2 for sd in subdomains)

    assert len(grouped_sources) == len(src_pos)
    grouped_ids = [s.id for s in grouped_sources]
    assert len(set(grouped_ids)) == len(src_pos)
    assert set(grouped_ids) == set(range(len(src_pos)))

    # Sources sit at cell centres, so the position of source i must still be
    # (src_pos[i] + 0.5) * cell_size after grouping.
    pos_by_id = {s.id: s.pos for s in grouped_sources}
    for i in range(len(src_pos)):
        np.testing.assert_allclose(
            pos_by_id[i], (src_pos[i].astype(float) + 0.5) * cell_size
        )
