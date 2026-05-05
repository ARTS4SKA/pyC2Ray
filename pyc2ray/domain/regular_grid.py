from __future__ import annotations
import numpy as np
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.sources import SourceGroup


# ============================================================================
# This file contains the RegularGrid class, which is a specific implementation 
# of the Grid interface for regular cubic grids.
# ============================================================================

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

    def __init__(self, cell_size: float, num_cells: int, offset: np.ndarray = np.array([0, 0, 0], dtype=np.int64), 
                 is_periodic_mode_active: bool = False) -> None:
        super().__init__(is_periodic_mode_active)
        self.cell_size = cell_size
        self.num_cells = num_cells
        self.offset = offset.copy()

    def get_domain_min(self) -> np.ndarray:
        """Get the space coordinates of the minimum corner of the domain.

        Returns
        -------
        np.ndarray
            The minimum corner of the domain (shape `(3,)`).
        """
        return self.offset * self.cell_size

    def get_domain_max(self) -> np.ndarray:
        """Get the space coordinates of the maximum corner of the domain.

        Returns
        -------
        np.ndarray
            The maximum corner of the domain (shape `(3,)`).
        """
        return (self.offset + self.num_cells) * self.cell_size

    def global_to_local_map(self, global_field: np.ndarray, local_field: np.ndarray) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.

        It is assumed that the size of the global grid is the one corresponding to the global_field. 
        
        Parameters
        ----------
        global_field : np.ndarray
            The field defined on the global grid to map.
        local_field : np.ndarray
            The field defined on the local grid initialized with the corresponding values from the global grid. 
            This is an I/O parameter whose size is determined by the local grid.
        """

        # TODO: add missing shape checks and generalize to vectorial fields
        target_shape = (self.num_cells, self.num_cells, self.num_cells)
        if local_field.shape != target_shape:
            try:
                local_field.resize(target_shape)
            except ValueError:
                raise ValueError("Unable to resize local_field to match local grid shape.")

        local_field.fill(0.0)  # Initialize local field to zero
        global_grid_size = global_field.shape[0]  # Assuming cubic grid

        if self.is_periodic_mode_active:

            # Build periodic index vectors for the full local cube. This handles
            # cells outside the global box by wrapping to the opposite side.
            gi = (np.arange(self.num_cells, dtype=np.int64) + self.offset[0]) % global_grid_size
            gj = (np.arange(self.num_cells, dtype=np.int64) + self.offset[1]) % global_grid_size
            gk = (np.arange(self.num_cells, dtype=np.int64) + self.offset[2]) % global_grid_size

            # Fill caller-provided local_field in place.
            local_field[:, :, :] = global_field[np.ix_(gi, gj, gk)]

        else:

            clipped_min = np.maximum(0, self.offset)
            clipped_max = np.minimum(self.offset + self.num_cells - 1, global_grid_size - 1)

            if np.any(clipped_min > clipped_max):
                raise ValueError("Local grid is completely outside the global grid, which is not supported.")

            local_offset = clipped_min - self.offset
            clipped_shape = clipped_max - clipped_min + 1
            local_field[
                local_offset[0]:local_offset[0] + clipped_shape[0],
                local_offset[1]:local_offset[1] + clipped_shape[1],
                local_offset[2]:local_offset[2] + clipped_shape[2],
            ] = global_field[
                clipped_min[0]:clipped_max[0] + 1,
                clipped_min[1]:clipped_max[1] + 1,
                clipped_min[2]:clipped_max[2] + 1,
            ]

    def local_to_global_map(self, local_field: np.ndarray, global_field: np.ndarray) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid
        and update the global field by adding the local field values.

        It is assumed that the size of the global grid is the one corresponding to the global_field. 
        
        Parameters
        ----------
        local_field : np.ndarray
            The field defined on the local grid to map.
        global_field : np.ndarray
            The field defined on the global grid to update with the local field values.
            This is an I/O parameter, which is updated in place with the values of the local field 
            corresponding to the grid elements included in the current grid.
        """
        # TODO: add missing shape checks and generalize to vectorial fields
        global_grid_size = global_field.shape[0]  # Assuming cubic grid

        if self.is_periodic_mode_active:
            for i_local in range(self.num_cells):
                for j_local in range(self.num_cells):
                    for k_local in range(self.num_cells):
                        i_global = (self.offset[0] + i_local) % global_grid_size
                        j_global = (self.offset[1] + j_local) % global_grid_size
                        k_global = (self.offset[2] + k_local) % global_grid_size
                        # TODO: make this behavior configurable
                        # += is needed when more groups are assigned to the same rank or 
                        # if subdomain is larger than the global domain (which should not happen in practice)
                        global_field[i_global, j_global, k_global] += local_field[i_local, j_local, k_local]

        else:
            clipped_min = np.maximum(0, self.offset)
            clipped_max = np.minimum(self.offset + self.num_cells - 1, global_grid_size - 1)
            if np.any(clipped_min > clipped_max):
                raise ValueError("Local grid is completely outside the global grid, which is not supported.")

            local_offset = clipped_min - self.offset
            clipped_shape = clipped_max - clipped_min + 1

            global_field[
                clipped_min[0]:clipped_max[0] + 1,
                clipped_min[1]:clipped_max[1] + 1,
                clipped_min[2]:clipped_max[2] + 1,
            ] = local_field[
                local_offset[0]:local_offset[0] + clipped_shape[0],
                local_offset[1]:local_offset[1] + clipped_shape[1],
                local_offset[2]:local_offset[2] + clipped_shape[2],
            ]


    def _overlap_volume(self, box_min, box_max) -> float:
        """Return intersection volume between two axis-aligned boxes.

        Parameters
        ----------
        box_min : np.ndarray
            Minimum corner of the box (shape `(3,)`).
        box_max : np.ndarray
            Maximum corner of the box (shape `(3,)`).

        Returns
        -------
        float
            Intersection volume between the two boxes.
        """
        if np.any(box_max <= box_min):
            raise ValueError('Invalid box: max corners must be "greater" than min corner.')
        overlap_min = np.maximum(self.offset*self.cell_size, box_min)
        overlap_max = np.minimum(self.offset*self.cell_size + self.num_cells*self.cell_size, box_max)
        d = np.maximum(0.0, overlap_max - overlap_min)
        return float(d[0] * d[1] * d[2])

    # TODO: the SourceGroup should not be needed here, the bounding box is enough. More in general, since
    # the region of influence of a source group could be defined in more complex ways than just a box, 
    # a specific class should be implemented to represent the region itself and passed to this method. 
    def get_local_grid(self, source_group: SourceGroup) -> RegularGrid:
        """Get the local grid corresponding to the region of influence of the source group of the subdomain.

        Parameters
        ----------
        source_group : SourceGroup
            The group of sources for which to get the local grid.

        Returns
        -------
        RegularGrid
            The local grid corresponding to the region of influence of the source group of the subdomain.
        """

        if np.any(source_group.bbox_max <= source_group.bbox_min):
            raise ValueError("Invalid box: box_max must be greater than box_min.")
    
        vol = self._overlap_volume(source_group.bbox_min, source_group.bbox_max)
        if vol > 0.0:
            # This calculation already accounts for the fact that the box may be partially outside the grid domain
            min_indexes = np.floor(source_group.bbox_min / self.cell_size).astype(int)
            max_indexes = np.ceil(source_group.bbox_max / self.cell_size).astype(int) - 1
            # Compute the effective volume contained in the grid domain
            # min_indexes_clipped = np.maximum(min_indexes, 0)
            # max_indexes_clipped = np.minimum(max_indexes, self.num_cells - 1)
            return RegularGrid(self.cell_size, max_indexes[0] - min_indexes[0] + 1, min_indexes, self.is_periodic_mode_active)
        else:
            # The box is completely outside the grid domain, return an empty grid
            return RegularGrid(self.cell_size, 0, np.array([0, 0, 0], dtype=np.int64), self.is_periodic_mode_active)

    # TODO: refactoring possible with get_local_grid
    def find_num_cells_in_box(self, box_min: np.ndarray, box_max: np.ndarray) -> int:
        """Find the number of cells in the box defined by the minimum and maximum corners.

        Parameters
        ----------
        box_min : np.ndarray
            The minimum corner of the box (shape `(3,)`).
        box_max : np.ndarray
            The maximum corner of the box (shape `(3,)`).

        Returns
        -------
        int
            The number of cells in the box defined by the minimum and maximum corners.
        """
        vol = self._overlap_volume(box_min, box_max)
        if vol > 0.0:
            min_indexes = np.floor(box_min / self.cell_size).astype(int)
            max_indexes = np.ceil(box_max / self.cell_size).astype(int) - 1
            if self.is_periodic_mode_active:
                # This calculation already accounts for the fact that the box may be partially outside the grid domain
                return int(np.prod((max_indexes - min_indexes + 1)))
            else:
                # In non-periodic mode, if the box extends outside the domain, we consider only the part inside the domain for the cell count
                min_indexes_clipped = np.maximum(min_indexes, self.offset)
                max_indexes_clipped = np.minimum(max_indexes, self.offset + self.num_cells - 1)
                return int(np.prod((max_indexes_clipped - min_indexes_clipped + 1)))
        else:
            return 0

    def get_total_num_cells(self) -> int:
        return self.num_cells ** 3