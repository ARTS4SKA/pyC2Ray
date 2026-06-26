"""
This file contains the RegularGrid class, which is a specific implementation
of the Grid interface for regular cubic grids.
"""

from __future__ import annotations

import numpy as np

from pyc2ray.domain.grid import Grid
from pyc2ray.domain.sources import SourceGroup


class RegularGrid(Grid):
    """Regular cubic grid implementation of the Grid interface.

    Attributes
    ----------
    cell_size : float
        Size of the grid cells (assuming cubic cells).
    num_cells : int
        Number of cells in each dimension (assuming cubic grid).
    offset : np.ndarray
        Offset (indexes) of the grid, relevant in case of sub-grids. For the global grid this is (0, 0, 0), so the minimum corner
        of the grid is at the origin. For a sub-grid, this is the index of the minimum corner of the local grid in the global grid.
        If the local grid is partially outside the global grid, this is the index of the minimum corner of the local grid,
        which may be negative if the local grid extends outside the global grid in the negative direction.
    """

    def __init__(
        self,
        cell_size: float,
        num_cells: int,
        offset: np.ndarray | None = None,
        is_periodic_mode_active: bool = False,
    ) -> None:
        super().__init__(is_periodic_mode_active)
        self.cell_size = cell_size
        self.num_cells = num_cells
        if offset is None:
            self.offset = np.array([0, 0, 0], dtype=np.int64)
        else:
            self.offset = np.asarray(offset, dtype=np.int64).copy()
            if self.offset.shape != (3,):
                raise ValueError("offset must have shape (3,)")

    def get_domain_min(self) -> np.ndarray:
        """Get the space coordinates of the minimum corner of the domain.

        Returns
        -------
        The minimum corner of the domain (shape `(3,)`).
        """
        return self.offset * self.cell_size

    def get_domain_max(self) -> np.ndarray:
        """Get the space coordinates of the maximum corner of the domain.

        Returns
        -------
        The maximum corner of the domain (shape `(3,)`).
        """
        return (self.offset + self.num_cells) * self.cell_size

    def get_average_cell_size(self, position: np.ndarray) -> float:
        """Get a representative (average) cell size of the grid in the neighbourhood of a position.

        For a regular grid the cell size is uniform, so the provided position is ignored and the
        constant cell size is returned.

        Parameters
        ----------
        position : The position in domain coordinates at which to evaluate the cell size (shape `(3,)`).

        Returns
        -------
        The (constant) cell size of the regular grid.
        """
        return self.cell_size

    def _periodic_global_indices(
        self, global_grid_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return wrapped global index vectors for each local-grid axis."""
        gijk = (
            np.arange(self.num_cells, dtype=np.int64) + self.offset[:, None]
        ) % global_grid_size
        return tuple(gijk)

    def _non_periodic_overlap_slices(
        self, global_grid_size: int
    ) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
        """Return local/global slices for the local grid portion inside the global grid."""
        if self.num_cells == 0:
            empty_slices = (slice(0, 0), slice(0, 0), slice(0, 0))
            return empty_slices, empty_slices

        clipped_min = np.maximum(0, self.offset)
        clipped_max = np.minimum(self.offset + self.num_cells - 1, global_grid_size - 1)

        if np.any(clipped_min > clipped_max):
            raise ValueError(
                "Local non-periodic grid is completely outside the global grid, which is not supported."
            )

        local_offset = clipped_min - self.offset
        clipped_shape = clipped_max - clipped_min + 1

        local_slices = (
            slice(local_offset[0], local_offset[0] + clipped_shape[0]),
            slice(local_offset[1], local_offset[1] + clipped_shape[1]),
            slice(local_offset[2], local_offset[2] + clipped_shape[2]),
        )
        global_slices = (
            slice(clipped_min[0], clipped_max[0] + 1),
            slice(clipped_min[1], clipped_max[1] + 1),
            slice(clipped_min[2], clipped_max[2] + 1),
        )

        return local_slices, global_slices

    def global_to_local_map(
        self, global_field: np.ndarray, local_field: np.ndarray
    ) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.

        It is assumed that the size of the global grid is the one corresponding to the global_field.

        Parameters
        ----------
        global_field : The field defined on the global grid to map.
        local_field : The field defined on the local grid initialized with the corresponding values
        from the global grid. This is an I/O parameter whose size is determined by the local grid.
        """

        # TODO: add missing shape checks and generalize to vectorial fields
        self.resize_local_field(local_field)

        global_grid_size = global_field.shape[0]  # Assuming cubic grid

        # TODO: add check, are we touching all the local grid cells in the assignements below?
        if self.is_periodic_mode_active:
            # Fill caller-provided local_field in place.
            local_field[:, :, :] = global_field[
                np.ix_(*self._periodic_global_indices(global_grid_size))
            ]

        else:
            # TODO: this is probably not needed anymore (check raytracing code) and should not be
            # here anyway, since it cretates a strong coupling between the mapping
            # and the specific value used to represent "no data" in the external code.
            local_field.fill(-1.0)

            local_slices, global_slices = self._non_periodic_overlap_slices(
                global_grid_size
            )
            local_field[local_slices] = global_field[global_slices]

    def local_to_global_map(
        self, local_field: np.ndarray, global_field: np.ndarray, add: bool = False
    ) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid
        and update the global field by adding the local field values.

        It is assumed that the size of the global grid is the one corresponding to the global_field.

        Parameters
        ----------
        local_field : The field defined on the local grid to map.
        global_field : The field defined on the global grid to update with the local field values.
        This is an I/O parameter, which is updated in place with the values of the local field
        corresponding to the grid elements included in the current grid.
        add : If True, the local field values are added to the global field values. If False, the local
        field values are set to the global field values.
        """
        # TODO: add missing shape checks and generalize to vectorial fields
        global_grid_size = global_field.shape[0]  # Assuming cubic grid

        if self.is_periodic_mode_active:
            global_indices = np.ix_(*self._periodic_global_indices(global_grid_size))
            if add:
                global_field[global_indices] += local_field
            else:
                global_field[global_indices] = local_field

        else:
            local_slices, global_slices = self._non_periodic_overlap_slices(
                global_grid_size
            )

            if add:
                global_field[global_slices] += local_field[local_slices]
            else:
                global_field[global_slices] = local_field[local_slices]

    def _validate_box(self, box_min: np.ndarray, box_max: np.ndarray) -> None:
        """Validate axis-aligned box corner ordering."""
        if np.any(box_max <= box_min):
            raise ValueError(
                'Invalid box: max corners must be "greater" than min corner.'
            )

    def _box_index_bounds(
        self, box_min: np.ndarray, box_max: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return inclusive cell-index bounds touched by a physical box."""
        self._validate_box(box_min, box_max)
        min_indexes = np.floor(box_min / self.cell_size).astype(np.int64)
        max_indexes = np.ceil(box_max / self.cell_size).astype(np.int64) - 1
        return min_indexes, max_indexes

    def _clip_index_bounds_to_grid(
        self, min_indexes: np.ndarray, max_indexes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip inclusive cell-index bounds to this grid."""
        clipped_min = np.maximum(min_indexes, self.offset)
        clipped_max = np.minimum(max_indexes, self.offset + self.num_cells - 1)
        return clipped_min, clipped_max

    def _overlap_volume(self, box_min: np.ndarray, box_max: np.ndarray) -> float:
        """Return intersection volume between two axis-aligned boxes.

        Parameters
        ----------
        box_min : Minimum corner of the box (shape `(3,)`).
        box_max : Maximum corner of the box (shape `(3,)`).

        Returns
        -------
        Intersection volume between the two boxes.
        """
        self._validate_box(box_min, box_max)
        overlap_min = np.maximum(self.offset * self.cell_size, box_min)
        overlap_max = np.minimum(
            self.offset * self.cell_size + self.num_cells * self.cell_size, box_max
        )
        d = np.maximum(0.0, overlap_max - overlap_min)
        return np.prod(d)

    # TODO: the SourceGroup should not be needed here, the bounding box is enough. More in general, since
    # the region of influence of a source group could be defined in more complex ways than just a box,
    # a specific class should be implemented to represent the region itself and passed to this method.
    def get_local_grid(self, source_group: SourceGroup) -> RegularGrid:
        """Get the local grid corresponding to the region of influence of the source group of the subdomain.

        Parameters
        ----------
        source_group : The group of sources for which to get the local grid.

        Returns
        -------
        The local grid corresponding to the region of influence of the source group of the subdomain.
        """

        min_indexes, max_indexes = self._box_index_bounds(
            source_group.bbox_min, source_group.bbox_max
        )
        vol = self._overlap_volume(source_group.bbox_min, source_group.bbox_max)
        if vol > 0.0:
            # This calculation already accounts for the fact that the box may be partially outside the grid domain
            side_lengths = max_indexes - min_indexes + 1
            # Conservative rounding policy: one extra cell is fine, one missing cell is not.
            num_cells = int(np.max(side_lengths))
            return RegularGrid(
                self.cell_size, num_cells, min_indexes, self.is_periodic_mode_active
            )

        # The box is completely outside the grid domain, return an empty grid
        return RegularGrid(
            self.cell_size,
            0,
            np.array([0, 0, 0], dtype=np.int64),
            self.is_periodic_mode_active,
        )

    def find_num_cells_in_box(self, box_min: np.ndarray, box_max: np.ndarray) -> int:
        """Find the number of cells in the box defined by the minimum and maximum corners.

        Parameters
        ----------
        box_min : The minimum corner of the box (shape `(3,)`).
        box_max : The maximum corner of the box (shape `(3,)`).

        Returns
        -------
        The number of cells in the box defined by the minimum and maximum corners.
        """
        min_indexes, max_indexes = self._box_index_bounds(box_min, box_max)
        vol = self._overlap_volume(box_min, box_max)
        if vol > 0.0:
            if self.is_periodic_mode_active:
                # This calculation already accounts for the fact that the box may be partially outside the grid domain
                return int(np.prod((max_indexes - min_indexes + 1)))

            # In non-periodic mode, if the box extends outside the domain, we consider only the part inside the domain for the cell count
            min_indexes_clipped, max_indexes_clipped = self._clip_index_bounds_to_grid(
                min_indexes, max_indexes
            )
            return int(np.prod((max_indexes_clipped - min_indexes_clipped + 1)))

        return 0

    def global_to_local_index_map(self, global_index: np.ndarray) -> np.ndarray:
        """Map a global grid index to the corresponding local grid index.

        Parameters
        ----------
        global_index : The global grid index to map (shape `(3,)`).

        Returns
        -------
        The corresponding local grid index (shape `(3,)`).
        """
        local_index = global_index - self.offset
        if np.any(local_index < 0) or np.any(local_index >= self.num_cells):
            raise ValueError("Global index is outside the local grid.")
        return local_index

    def global_to_local_position_map(self, global_position: np.ndarray) -> np.ndarray:
        """Map a global position in domain coordinates to the corresponding local subdomain coordinates.

        Parameters
        ----------
        global_position : The global position in domain coordinates to map (shape `(3,)`).

        Returns
        -------
        The corresponding local grid index (shape `(3,)`).
        """
        global_index = np.floor(global_position / self.cell_size).astype(int)
        return self.global_to_local_index_map(global_index)

    def resize_local_field(self, local_field: np.ndarray) -> None:
        """Resize a local field to match the size of the local grid.

        Parameters
        ----------
        local_field : The local field to resize (IO parameter).
        """
        target_shape = (self.num_cells, self.num_cells, self.num_cells)
        if local_field.shape != target_shape:
            try:
                local_field.resize(target_shape)
            except ValueError as e:
                raise ValueError(
                    f"Unable to resize local_field to match local grid shape: {e}"
                )
