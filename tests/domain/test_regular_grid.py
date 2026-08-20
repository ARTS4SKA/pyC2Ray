"""Unit tests for domain decomposition (regular) grid class."""

from __future__ import annotations

import numpy as np
import pytest

from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source, SourceGroup


def test_regular_grid_domain_min_max_with_offset() -> None:
    grid = RegularGrid(
        cell_size=2.0, num_cells=4, offset=np.array([1, 2, 3], dtype=np.int64)
    )

    np.testing.assert_allclose(grid.get_domain_min(), np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(grid.get_domain_max(), np.array([10.0, 12.0, 14.0]))


def test_regular_grid_global_to_local_map_non_periodic() -> None:
    # Global field defined on a 5x5x5 grid
    global_size = 5
    global_field = np.arange(global_size**3, dtype=float).reshape(
        global_size, global_size, global_size
    )

    # Local grid defined with offset (1, 0, 1) and size 4x4x4 partially outside the global grid
    local_size = 4
    grid = RegularGrid(
        cell_size=1.0, num_cells=local_size, offset=np.array([-1, 0, 1], dtype=np.int64)
    )
    local_field = np.empty((1, 1, 1), dtype=float)

    expected_local_field = np.full(
        (local_size, local_size, local_size), -1.0, dtype=float
    )
    expected_local_field[1:, :, :] = global_field[0:3, 0:4, 1:5]

    grid.global_to_local_map(global_field, local_field)

    assert local_field.shape == (local_size, local_size, local_size)
    np.testing.assert_allclose(local_field, expected_local_field)


def test_regular_grid_global_to_local_map_non_periodic_included() -> None:
    # Global field defined on a 5x5x5 grid
    global_size = 5
    global_field = np.arange(global_size**3, dtype=float).reshape(
        global_size, global_size, global_size
    )

    # Local grid defined with offset (1, 1, 1) and size 4x4x4, fully included in the global grid
    local_size = 4
    grid = RegularGrid(
        cell_size=1.0, num_cells=local_size, offset=np.array([1, 1, 1], dtype=np.int64)
    )
    local_field = np.empty((1, 1, 1), dtype=float)

    expected_local_field = np.full(
        (local_size, local_size, local_size), -1.0, dtype=float
    )
    expected_local_field[:, :, :] = global_field[1:5, 1:5, 1:5]

    grid.global_to_local_map(global_field, local_field)

    assert local_field.shape == (local_size, local_size, local_size)
    np.testing.assert_allclose(local_field, expected_local_field)


def test_regular_grid_global_to_local_map_periodic() -> None:
    # Global field defined on a 4x4x4 grid
    global_size = 4
    global_field = np.arange(global_size**3, dtype=float).reshape(
        global_size, global_size, global_size
    )

    # Local periodic grid defined with offset (3, -1, 4) and size 2x2x2
    local_size = 2
    grid = RegularGrid(
        cell_size=1.0,
        num_cells=local_size,
        offset=np.array([3, -1, 4], dtype=np.int64),
        is_periodic_mode_active=True,
    )
    local_field = np.empty((local_size, local_size, local_size), dtype=float)

    expected_local_field = global_field[np.ix_([3, 0], [3, 0], [0, 1])]

    grid.global_to_local_map(global_field, local_field)

    assert local_field.shape == (local_size, local_size, local_size)
    np.testing.assert_allclose(local_field, expected_local_field)


def test_regular_grid_global_to_local_map_periodic_included() -> None:
    # Global field defined on a 4x4x4 grid
    global_size = 4
    global_field = np.arange(global_size**3, dtype=float).reshape(
        global_size, global_size, global_size
    )

    # Local periodic grid defined with offset (0, 1, 0) and size 2x2x2, fully included in the global grid
    local_size = 2
    grid = RegularGrid(
        cell_size=1.0,
        num_cells=local_size,
        offset=np.array([0, 1, 0], dtype=np.int64),
        is_periodic_mode_active=True,
    )
    local_field = np.empty((local_size, local_size, local_size), dtype=float)

    expected_local_field = global_field[np.ix_([0, 1], [1, 2], [0, 1])]

    grid.global_to_local_map(global_field, local_field)

    assert local_field.shape == (local_size, local_size, local_size)
    np.testing.assert_allclose(local_field, expected_local_field)


def test_regular_grid_local_to_global_map_non_periodic_set_and_add() -> None:
    grid = RegularGrid(
        cell_size=1.0, num_cells=3, offset=np.array([3, 1, 0], dtype=np.int64)
    )
    local_field = np.ones((3, 3, 3), dtype=float)
    global_field = np.zeros((5, 5, 5), dtype=float)

    grid.local_to_global_map(local_field, global_field, add=False)
    assert np.all(global_field[3:6, 1:4, 0:3] == 1.0)
    assert np.all(global_field[0:3, 0:1, 4:6] == 0.0)

    grid.local_to_global_map(local_field, global_field, add=True)
    assert np.all(global_field[3:6, 1:4, 0:3] == 2.0)
    assert np.all(global_field[0:3, 0:1, 4:6] == 0.0)


def test_regular_grid_local_to_global_map_non_periodic_set_and_add_included() -> None:
    grid = RegularGrid(
        cell_size=1.0, num_cells=3, offset=np.array([1, 1, 1], dtype=np.int64)
    )
    local_field = np.ones((3, 3, 3), dtype=float)
    global_field = np.zeros((5, 5, 5), dtype=float)

    grid.local_to_global_map(local_field, global_field, add=False)
    assert np.all(global_field[1:4, 1:4, 1:4] == 1.0)
    assert np.all(global_field[0:1, 0:1, 0:1] == 0.0)

    grid.local_to_global_map(local_field, global_field, add=True)
    assert np.all(global_field[1:4, 1:4, 1:4] == 2.0)
    assert np.all(global_field[0:1, 0:1, 0:1] == 0.0)


def test_regular_grid_local_to_global_map_periodic_set_and_add() -> None:
    grid = RegularGrid(
        cell_size=1.0,
        num_cells=2,
        offset=np.array([3, 3, 3], dtype=np.int64),
        is_periodic_mode_active=True,
    )
    local_field = np.ones((2, 2, 2), dtype=float)
    global_field = np.zeros((4, 4, 4), dtype=float)

    expected_global_field = np.zeros((4, 4, 4), dtype=float)
    expected_global_field[0, 0, 0] = 1
    expected_global_field[3, 0, 0] = 1
    expected_global_field[3, 3, 0] = 1
    expected_global_field[0, 3, 0] = 1
    expected_global_field[0, 0, 3] = 1
    expected_global_field[0, 3, 3] = 1
    expected_global_field[3, 0, 3] = 1
    expected_global_field[3, 3, 3] = 1
    grid.local_to_global_map(local_field, global_field, add=False)
    assert np.all(global_field == expected_global_field)

    expected_global_field *= 2
    grid.local_to_global_map(local_field, global_field, add=True)
    assert np.all(global_field == expected_global_field)


def test_regular_grid_periodic_mapping_rejects_duplicate_wrapping() -> None:
    grid = RegularGrid(
        cell_size=1.0,
        num_cells=5,
        offset=np.array([3, 3, 3], dtype=np.int64),
        is_periodic_mode_active=True,
    )
    local_field = np.ones((5, 5, 5), dtype=float)
    global_field = np.zeros((4, 4, 4), dtype=float)

    with pytest.raises(ValueError, match="multiple local cells"):
        grid.local_to_global_map(local_field, global_field, add=True)


def test_regular_grid_local_to_global_map_periodic_set_and_add_included() -> None:
    grid = RegularGrid(
        cell_size=1.0,
        num_cells=2,
        offset=np.array([0, 0, 0], dtype=np.int64),
        is_periodic_mode_active=True,
    )
    local_field = np.ones((2, 2, 2), dtype=float)
    global_field = np.zeros((4, 4, 4), dtype=float)

    expected_global_field = np.zeros((4, 4, 4), dtype=float)
    expected_global_field[0:2, 0:2, 0:2] = 1
    grid.local_to_global_map(local_field, global_field, add=False)
    assert np.all(global_field == expected_global_field)

    expected_global_field[0:2, 0:2, 0:2] = 2
    grid.local_to_global_map(local_field, global_field, add=True)
    assert np.all(global_field == expected_global_field)


def test_regular_grid_find_num_cells_in_box() -> None:
    box_min = np.array([-1.2, 0.2, 0.2])
    box_max = np.array([2.2, 3.2, 3.2])

    non_periodic = RegularGrid(cell_size=1.0, num_cells=4)
    periodic = RegularGrid(cell_size=1.0, num_cells=4, is_periodic_mode_active=True)

    # The box spans 5 cells on the first axis, i.e. more than the grid: in periodic mode it wraps
    # onto itself and still touches only the 4 distinct cells of that axis, hence 4 * 4 * 4.
    assert periodic.find_num_cells_in_box(box_min, box_max) == 64
    assert non_periodic.find_num_cells_in_box(box_min, box_max) == 48


def test_regular_grid_get_local_grid_from_source_group_included() -> None:
    global_grid = RegularGrid(cell_size=1.0, num_cells=10)
    src_pos = np.array([1.0, 1.0, 1.0])
    src = Source(id=1, pos=src_pos, strength=1.0, radius=1.0)
    group = SourceGroup(
        id=0,
        sources=[src],
        center=src_pos,
        radius=1.0,
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([2.0, 2.0, 2.0]),
        mem_cost=1.0,
        comp_cost=1.0,
    )

    local_grid = global_grid.get_local_grid(group)

    np.testing.assert_array_equal(local_grid.offset, np.array([0, 0, 0]))
    assert local_grid.num_cells == 2


def test_regular_grid_get_local_grid_from_source_group() -> None:
    global_grid = RegularGrid(cell_size=1.0, num_cells=10)
    src_pos = np.array([9.0, 9.0, 9.0])
    src = Source(id=1, pos=src_pos, strength=1.0, radius=1.5)
    group = SourceGroup(
        id=0,
        sources=[src],
        center=src_pos,
        radius=1.5,
        bbox_min=np.array([7.5, 7.5, 7.5]),
        bbox_max=np.array([10.5, 10.5, 10.5]),
        mem_cost=1.0,
        comp_cost=1.0,
    )

    local_grid = global_grid.get_local_grid(group)

    np.testing.assert_array_equal(local_grid.offset, np.array([7, 7, 7]))
    assert local_grid.num_cells == 4


def test_regular_grid_get_local_grid_larger_than_domain() -> None:
    """Test the case of a source group with a bounding box larger that the global grid."""
    global_grid = RegularGrid(cell_size=1.0, num_cells=4, is_periodic_mode_active=True)
    src_pos = np.array([2.0, 2.0, 2.0])
    src = Source(id=1, pos=src_pos, strength=1.0, radius=20.0)
    group = SourceGroup(
        id=0,
        sources=[src],
        center=src_pos,
        radius=20.0,
        bbox_min=src_pos - 20.0,
        bbox_max=src_pos + 20.0,
        mem_cost=1.0,
        comp_cost=1.0,
    )

    local_grid = global_grid.get_local_grid(group)

    assert local_grid.num_cells == global_grid.num_cells
    np.testing.assert_array_equal(local_grid.offset, np.array([0, 0, 0]))

    # The capped local grid must be usable by the periodic mapping, which forbids local grids
    # larger than the global one.
    global_field = np.arange(4**3, dtype=float).reshape((4, 4, 4))
    local_field = np.empty((0,), dtype=float)
    local_grid.resize_local_field(local_field)
    local_grid.global_to_local_map(global_field, local_field)
    np.testing.assert_allclose(local_field, global_field)


def test_regular_grid_get_local_grid_outside_domain_is_empty() -> None:
    global_grid = RegularGrid(cell_size=1.0, num_cells=4)
    src_pos = np.array([10.0, 10.0, 10.0])
    src = Source(id=1, pos=src_pos, strength=1.0, radius=1.2)
    group = SourceGroup(
        id=0,
        sources=[src],
        center=src_pos,
        radius=1.2,
        bbox_min=np.array([8.0, 8.0, 8.0]),
        bbox_max=np.array([12.0, 12.0, 12.0]),
        mem_cost=1.0,
        comp_cost=1.0,
    )

    local_grid = global_grid.get_local_grid(group)

    assert local_grid.num_cells == 0
    np.testing.assert_array_equal(local_grid.offset, np.array([0, 0, 0]))


def test_regular_grid_mapping_with_empty_local_grid() -> None:
    grid = RegularGrid(cell_size=1.0, num_cells=0)
    global_field = np.ones((4, 4, 4), dtype=float)
    local_field = np.empty((1, 1, 1), dtype=float)

    grid.global_to_local_map(global_field, local_field)

    assert local_field.shape == (0, 0, 0)

    expected_global_field = global_field.copy()
    grid.local_to_global_map(local_field, global_field, add=False)
    np.testing.assert_allclose(global_field, expected_global_field)

    grid.local_to_global_map(local_field, global_field, add=True)
    np.testing.assert_allclose(global_field, expected_global_field)


def test_regular_grid_index_and_position_maps() -> None:
    grid = RegularGrid(
        cell_size=0.5, num_cells=4, offset=np.array([2, 2, 2], dtype=np.int64)
    )

    np.testing.assert_array_equal(
        grid.global_to_local_index_map(np.array([3, 4, 5])), np.array([1, 2, 3])
    )
    np.testing.assert_array_equal(
        grid.global_to_local_position_map(np.array([1.25, 1.75, 2.75])),
        np.array([0, 1, 3]),
    )

    with pytest.raises(ValueError):
        grid.global_to_local_index_map(np.array([1, 2, 2]))


def test_regular_grid_resize_local_field() -> None:
    grid = RegularGrid(cell_size=1.0, num_cells=3)
    local_field = np.zeros((2, 2, 2), dtype=float)

    grid.resize_local_field(local_field)
    assert local_field.shape == (3, 3, 3)


def test_regular_grid_overlap_volume_invalid_box_raises() -> None:
    grid = RegularGrid(cell_size=1.0, num_cells=4)

    with pytest.raises(ValueError):
        grid._overlap_volume(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 2.0]))
