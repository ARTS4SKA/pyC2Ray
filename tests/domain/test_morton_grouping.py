"""Unit tests for domain decomposition Morton grouping class."""

from __future__ import annotations

import numpy as np
import pytest

from pyc2ray.domain.cost_model import CostModel, pyC2RayCostModel
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


class CellCountCostModel(CostModel):
    """Cost model whose memory cost is the number of cells of the local grid.

    Unlike DummyCostModel it depends on the group geometry, so it exposes the
    effect of the cell discretization of the enclosing sphere on the cost.
    """

    def compute_group_costs(
        self, R: float, n_cells_per_side: int, n_src: int
    ) -> tuple[float, float]:
        return float(n_cells_per_side**3), float(n_src)


def _src(sid: int, x: float, y: float, z: float, radius: float = 0.25) -> Source:
    return Source(
        id=sid, pos=np.array([x, y, z], dtype=float), strength=1.0, radius=radius
    )


def test_key_increases_along_axes() -> None:
    """Test that the Morton key increases when moving along each axis from the origin."""

    grouping = MortonSourceGrouping()
    domain_min = np.array([0.0, 0.0, 0.0], dtype=float)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=float)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.8],
        ],
        dtype=float,
    )

    key_origin, key_x, key_y, key_z = grouping._morton_like_keys(
        positions, domain_min, domain_max, bits=8
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
    positions = np.array(
        [
            [0.20, 0.20, 0.20],  # reference
            [0.21, 0.20, 0.20],  # near
            [0.85, 0.85, 0.85],  # far
        ],
        dtype=float,
    )

    key_ref, key_near, key_far = grouping._morton_like_keys(
        positions, domain_min, domain_max, bits=12
    )

    assert abs(key_ref - key_near) < abs(key_ref - key_far)


def test_keys_interleave_coordinate_bits() -> None:
    """Test the Morton keys against hand-computed values.

    Coordinate bit i of x, y and z must land at key bit 3*i, 3*i + 1 and 3*i + 2
    respectively. With bits=2 on a domain of side 4, each position maps to the
    integer cell (floor(x), floor(y), floor(z)).
    """
    grouping = MortonSourceGrouping()
    domain_min = np.array([0.0, 0.0, 0.0], dtype=float)
    domain_max = np.array([4.0, 4.0, 4.0], dtype=float)
    positions = np.array(
        [
            [0.5, 0.5, 0.5],  # cell (0, 0, 0)
            [1.5, 0.5, 0.5],  # cell (1, 0, 0): x bit 0 -> key bit 0
            [0.5, 1.5, 0.5],  # cell (0, 1, 0): y bit 0 -> key bit 1
            [0.5, 0.5, 1.5],  # cell (0, 0, 1): z bit 0 -> key bit 2
            [1.5, 1.5, 1.5],  # cell (1, 1, 1): key bits 0-2
            [2.5, 0.5, 0.5],  # cell (2, 0, 0): x bit 1 -> key bit 3
            [0.5, 2.5, 0.5],  # cell (0, 2, 0): y bit 1 -> key bit 4
            [0.5, 0.5, 2.5],  # cell (0, 0, 2): z bit 1 -> key bit 5
            [3.5, 3.5, 3.5],  # cell (3, 3, 3): all 6 key bits
            [3.5, 0.5, 2.5],  # cell (3, 0, 2): key bits 0, 3 and 5
        ],
        dtype=float,
    )
    expected = [0, 1, 2, 4, 7, 8, 16, 32, 63, 41]

    keys = grouping._morton_like_keys(positions, domain_min, domain_max, bits=2)

    assert keys.tolist() == expected


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


def test_build_groups_incremental_rejects_wrong_grouping_params_type() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)
    sources = [_src(1, 1.0, 1.0, 1.0)]

    with pytest.raises(TypeError):
        grouping.build_groups_incremental(
            sources=sources,
            grid=grid,
            grouping_params=GroupingParams(max_num_sources_per_group=2),
            cost_model=cost_model,
        )


def test_build_groups_incremental_returns_empty_for_no_sources() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=8)
    params = MortonGroupingParams(max_num_sources_per_group=4, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)

    groups = grouping.build_groups_incremental([], grid, params, cost_model)

    assert groups == []


def test_build_groups_incremental_single_valid_group() -> None:
    """Three nearby sources whose spheres intersect stay in one group."""
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=8)
    params = MortonGroupingParams(max_num_sources_per_group=4, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)
    sources = [
        _src(10, 1.0, 1.0, 1.0, radius=0.5),
        _src(11, 1.3, 1.1, 1.0, radius=0.5),
        _src(12, 1.6, 1.0, 1.1, radius=0.5),
    ]

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    assert len(groups) == 1
    assert groups[0].id == 0
    assert len(groups[0]) == 3
    assert sorted(groups[0].get_source_ids()) == [10, 11, 12]


def test_build_groups_incremental_splits_when_sources_do_not_intersect() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=8, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=10.0)
    sources = [
        _src(1, 1.0, 1.0, 1.0, radius=0.2),
        _src(2, 8.0, 8.0, 8.0, radius=0.2),
    ]

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    assert len(groups) == 2
    assert [g.id for g in groups] == [0, 1]
    assert sorted(len(g) for g in groups) == [1, 1]


def test_build_groups_incremental_splits_when_memory_limit_exceeded() -> None:
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=8, morton_bits=8)
    # With DummyCostModel, mem_cost == n_src; setting max to 1 forces a split
    # when trying to merge 2 intersecting sources.
    cost_model = DummyCostModel(max_memory_cost_per_group=1.0)
    sources = [
        _src(20, 2.0, 2.0, 2.0, radius=0.5),
        _src(21, 2.4, 2.0, 2.0, radius=0.5),
    ]

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    assert len(groups) == 2
    assert [g.id for g in groups] == [0, 1]
    assert all(len(g) == 1 for g in groups)


def test_build_groups_incremental_splits_when_source_cap_exceeded() -> None:
    """The source-count cap closes a group even when everything intersects."""
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=2, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=1e9)
    sources = [_src(30 + i, 2.0 + 0.1 * i, 2.0, 2.0, radius=1.0) for i in range(5)]

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    assert all(len(g) <= 2 for g in groups)
    assert sum(len(g) for g in groups) == 5


def test_build_groups_incremental_groups_enclose_their_sources() -> None:
    """Every group sphere must contain the influence sphere of each of its members.

    This is the property the incremental sphere has to preserve: the scan takes its
    decisions on an upper bound, and the group is re-fitted when closed, so a member
    poking out would mean the re-fit dropped below the bound it was checked against.
    """
    rng = np.random.default_rng(4321)
    positions = rng.uniform(1.0, 15.0, size=(120, 3))
    sources = [
        _src(i, positions[i][0], positions[i][1], positions[i][2], radius=0.8)
        for i in range(len(positions))
    ]
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=16, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=1e9)

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    for group in groups:
        for source in group.sources:
            reach = float(np.linalg.norm(group.center - source.pos)) + source.radius
            assert reach <= group.radius + 1e-9


def test_build_groups_incremental_conserves_every_source_exactly_once() -> None:
    """No source may be lost or duplicated when sources are split into groups."""
    rng = np.random.default_rng(99)
    positions = rng.uniform(1.0, 15.0, size=(200, 3))
    sources = [
        _src(i, positions[i][0], positions[i][1], positions[i][2], radius=0.6)
        for i in range(len(positions))
    ]
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=8, morton_bits=8)
    cost_model = DummyCostModel(max_memory_cost_per_group=1e9)

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    # Guard against a vacuous pass: the sources must actually have been split into
    # several groups, and at least one group must hold more than a single source.
    assert len(groups) > 1
    assert max(len(g) for g in groups) > 1
    assert all(len(g) <= 8 for g in groups)

    grouped_ids = [sid for g in groups for sid in g.get_source_ids()]
    assert sorted(grouped_ids) == sorted(s.id for s in sources)
    assert [g.id for g in groups] == list(range(len(groups)))


def test_build_groups_incremental_refit_respects_accepted_memory_cost() -> None:
    """The final re-fit must not push a group over the memory cap it was accepted under.

    For these sources the incremental sphere spans 5 cells per side, which is exactly
    the cap. The accurate fit has a radius smaller by less than 1e-4, but its center is
    shifted so that its box crosses one more cell boundary, giving 6 cells per side.
    Keeping the tighter sphere unconditionally would store a group of 6**3 cells.
    """
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=16)
    params = MortonGroupingParams(max_num_sources_per_group=8, morton_bits=8)
    cost_model = CellCountCostModel(max_memory_cost_per_group=5.0**3)
    sources = [
        _src(0, 7.9, 9.75, 5.19, radius=1.5),
        _src(1, 8.9, 8.92, 4.86, radius=1.5),
        _src(2, 8.22, 9.62, 4.83, radius=1.5),
        _src(3, 8.75, 8.95, 4.36, radius=1.5),
    ]

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    assert len(groups) == 1
    assert len(groups[0]) == 4
    assert groups[0].mem_cost <= cost_model.max_memory_cost_per_group
    # The stored geometry must actually be the one the cost was evaluated on.
    n_cells = grid.find_num_cells_in_box(groups[0].bbox_min, groups[0].bbox_max)
    assert int(np.ceil(n_cells ** (1.0 / 3.0))) ** 3 == groups[0].mem_cost


def test_build_groups_incremental_stored_costs_match_stored_geometry() -> None:
    """The costs reused from the scan must be the costs of the geometry stored on the group.

    The incremental scan passes the costs it already evaluated to the closed group
    instead of recomputing them, so a mismatch between the reused costs and the stored
    sphere would go unnoticed without this check.
    """
    rng = np.random.default_rng(2024)
    positions = rng.uniform(1.0, 31.0, size=(300, 3))
    sources = [
        _src(i, positions[i][0], positions[i][1], positions[i][2], radius=1.7)
        for i in range(len(positions))
    ]
    grouping = MortonSourceGrouping()
    grid: Grid = RegularGrid(cell_size=1.0, num_cells=32, is_periodic_mode_active=True)
    params = MortonGroupingParams(max_num_sources_per_group=12, morton_bits=8)
    cost_model = pyC2RayCostModel(
        max_memory_cost_per_group=2.0e4,
        source_batch_size=4,
        is_periodic_mode_active=True,
        photo_ion_table_size=16,
    )

    groups = grouping.build_groups_incremental(sources, grid, params, cost_model)

    # Without the following check, the test could pass without checking anything
    assert max(len(g) for g in groups) > 1
    # Guard against a vacuous pass: some groups must have gone through the re-fit.
    for group in groups:
        expected = grouping._group_costs(
            group.center,
            group.radius,
            group.sources[0].radius,
            len(group),
            grid,
            cost_model,
        )
        assert (group.mem_cost, group.comp_cost) == expected
        np.testing.assert_array_equal(group.bbox_min, group.center - group.radius)
        np.testing.assert_array_equal(group.bbox_max, group.center + group.radius)
        if len(group) > 1:
            assert group.mem_cost <= cost_model.max_memory_cost_per_group
