"""Unit tests for domain decomposition Morton grouping class."""

from __future__ import annotations

import numpy as np
import pytest

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.morton_grouping import MortonGroupingParams, MortonSourceGrouping
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.source_grouping import GroupingParams
from pyc2ray.domain.sources import Source


class DummyCostModel(CostModel):
    """Simple deterministic cost model for grouping tests."""

    def compute_group_costs(
        self, R: float, n_cells_per_side: int, n_src: int
    ) -> tuple[float, float]:
        # Keep memory cost controlled by number of sources so tests can
        # trigger acceptance/rejection paths deterministically.
        return float(n_src), float(n_src * n_cells_per_side)


def _src(sid: int, x: float, y: float, z: float, radius: float = 0.25) -> Source:
    return Source(
        id=sid, pos=np.array([x, y, z], dtype=float), strength=1.0, radius=radius
    )


def test_key_increases_along_axes() -> None:
    """Test that the Morton key increases when moving along each axis from the origin."""

    grouping = MortonSourceGrouping()
    domain_min = np.array([0.0, 0.0, 0.0], dtype=float)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=float)

    key_origin = grouping._morton_like_key(
        np.array([0.0, 0.0, 0.0]), domain_min, domain_max, bits=8
    )
    key_x = grouping._morton_like_key(
        np.array([0.8, 0.0, 0.0]), domain_min, domain_max, bits=8
    )
    key_y = grouping._morton_like_key(
        np.array([0.0, 0.8, 0.0]), domain_min, domain_max, bits=8
    )
    key_z = grouping._morton_like_key(
        np.array([0.0, 0.0, 0.8]), domain_min, domain_max, bits=8
    )

    assert key_origin < key_x
    assert key_origin < key_y
    assert key_origin < key_z


def test_points_have_nearer_keys_than_far_points() -> None:
    """Points closer in space should have closer Morton keys than points farther apart."""
    grouping = MortonSourceGrouping()
    domain_min = np.array([0.0, 0.0, 0.0], dtype=float)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=float)

    # Two points in the same neighborhood and one far-away point.
    p_ref = np.array([0.20, 0.20, 0.20], dtype=float)
    p_near = np.array([0.21, 0.20, 0.20], dtype=float)
    p_far = np.array([0.85, 0.85, 0.85], dtype=float)

    key_ref = grouping._morton_like_key(p_ref, domain_min, domain_max, bits=12)
    key_near = grouping._morton_like_key(p_near, domain_min, domain_max, bits=12)
    key_far = grouping._morton_like_key(p_far, domain_min, domain_max, bits=12)

    assert abs(key_ref - key_near) < abs(key_ref - key_far)


def test_build_groups_rejects_wrong_grouping_params_type() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)
    sources = [_src(1, 1.0, 1.0, 1.0)]

    with pytest.raises(TypeError):
        grouping.build_groups(
            sources=sources,
            grid=grid,
            grouping_params=GroupingParams(max_num_sources_per_group=2),
            cost_model=cost_model,
        )


def test_build_groups_returns_empty_for_no_sources() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=8)
    params = MortonGroupingParams(max_num_sources_per_group=4, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)

    groups = grouping.build_groups([], grid, params, cost_model)

    assert groups == []


def test_build_groups_single_valid_group() -> None:
    """Test that three nearby sources are grouped together when their spheres of influence
    intersect and memory cost is acceptable.
    """
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=8)
    params = MortonGroupingParams(max_num_sources_per_group=4, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)
    # Intersecting source spheres to allow grouping.
    sources = [
        _src(10, 1.0, 1.0, 1.0, radius=0.5),
        _src(11, 1.3, 1.1, 1.0, radius=0.5),
        _src(12, 1.6, 1.0, 1.1, radius=0.5),
    ]

    groups = grouping.build_groups(sources, grid, params, cost_model)

    assert len(groups) == 1
    assert groups[0].id == 0
    assert len(groups[0]) == 3
    assert sorted(groups[0].get_source_ids()) == [10, 11, 12]


def test_build_groups_splits_when_sources_do_not_intersect() -> None:
    """Test that two sources that do not have intersecting spheres of influence
    are not grouped together, even if memory cost is acceptable.
    """
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=8, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)
    # Far apart so each new source starts a new group.
    sources = [
        _src(1, 1.0, 1.0, 1.0, radius=0.2),
        _src(2, 8.0, 8.0, 8.0, radius=0.2),
    ]

    groups = grouping.build_groups(sources, grid, params, cost_model)

    assert len(groups) == 2
    assert [g.id for g in groups] == [0, 1]
    assert sorted(len(g) for g in groups) == [1, 1]


def test_build_groups_splits_when_memory_limit_exceeded() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=8, morton_bits=8)
    # With DummyCostModel, mem_cost == n_src; setting max to 1 forces split
    # when trying to merge 2 intersecting sources.
    cost_model = DummyCostModel(max_memory_cost_per_group=1.0)
    sources = [
        _src(20, 2.0, 2.0, 2.0, radius=0.5),
        _src(21, 2.4, 2.0, 2.0, radius=0.5),
    ]

    groups = grouping.build_groups(sources, grid, params, cost_model)

    assert len(groups) == 2
    assert [g.id for g in groups] == [0, 1]
    assert all(len(g) == 1 for g in groups)
