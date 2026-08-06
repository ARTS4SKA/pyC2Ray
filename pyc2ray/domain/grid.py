"""
This file provides an interface for the Grid class, which is used to
represent the full grid of the simulation and the sub-grids corresponding
to the subdomains. The idea is to have a common interface for different
grid implementations, such as regular grids, octrees, etc. The Grid class
should provide functionalities for handling the grid data, particularly
the mapping between physical coordinates and grid indexes and between
local subdomain grid indexes and global grid indexes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pyc2ray.domain.sources import SourceGroup


class Grid(ABC):
    """Base class for grid representation, providing a common interface for different
       grid implementations.

    Attributes
    ----------
    is_periodic_mode_active : Flag indicating whether periodic boundary conditions are
    active in the grid.
    """

    def __init__(self, is_periodic_mode_active: bool = False) -> None:
        self.is_periodic_mode_active = is_periodic_mode_active

    @abstractmethod
    def get_domain_min(self) -> np.ndarray:
        """Get the space coordinates of the minimum corner of the domain.

        Returns
        -------
        The minimum corner of the domain (shape `(3,)`).
        """

    @abstractmethod
    def get_domain_max(self) -> np.ndarray:
        """Get the space coordinates of the maximum corner of the domain.

        Returns
        -------
        The maximum corner of the domain (shape `(3,)`).
        """

    @abstractmethod
    def get_average_cell_size(self, position: np.ndarray) -> float:
        """Get a representative (average) cell size of the grid in the neighbourhood of a position.

        This is meant to convert physical lengths (e.g. a source radius of influence) into a number
        of grid cells. For uniform grids the result is constant; for non-uniform grids (e.g. adaptive
        mesh refinement) it depends on the local resolution around the provided position.

        Parameters
        ----------
        position : The position in domain coordinates at which to evaluate the cell size (shape `(3,)`).

        Returns
        -------
        The representative cell size of the grid around the provided position.
        """

    @abstractmethod
    def global_to_local_map(
        self, global_field: np.ndarray, local_field: np.ndarray
    ) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        global_field : The field defined on the global grid to map.
        local_field : The field defined on the local grid initialized with the corresponding values
        from the global grid. This is an I/O parameter.
        """

    @abstractmethod
    def local_to_global_map(
        self, local_field: np.ndarray, global_field: np.ndarray, add: bool = False
    ) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid.

        Parameters
        ----------
        local_field : The field defined on the local grid to map.
        global_field : The field defined on the global grid to update with the local field values.
        This is an I/O parameter, which is updated in place with the values of the local field
        corresponding to the grid elements included in the current grid.
        add : If True, the local field values are added to the global field values. If False, the local
        field values are set to the global field values.
        """

    # TODO: design issue: this function implicitly assumes that the local grid
    # is a subset of the global grid, which is the case for regular grids but
    # may not be the case for more general grid types.
    # We may need to rethink this interface if we want to support more general
    # grid types in the future.
    @abstractmethod
    def global_to_local_index_map(self, global_index: np.ndarray) -> np.ndarray:
        """Map a global grid index to the corresponding local grid index.

        Parameters
        ----------
        global_index : The global grid index to map (shape `(3,)`).

        Returns
        -------
        The corresponding local grid index (shape `(3,)`).
        """

    @abstractmethod
    def global_to_local_position_map(self, global_position: np.ndarray) -> np.ndarray:
        """Map a global position in domain coordinates to the corresponding local subdomain coordinates.

        Parameters
        ----------
        global_position : The global position in domain coordinates to map (shape `(3,)`).

        Returns
        -------
        The corresponding local subdomain coordinates (shape `(3,)`).
            The corresponding local grid index (shape `(3,)`).
        """

    @abstractmethod
    def get_local_grid(self, source_group: SourceGroup) -> Grid:
        """Get the local grid corresponding to the region of influence of the source group of the subdomain.

        Parameters
        ----------
        source_group : The group of sources for which to get the local grid.

        Returns
        -------
        The local grid corresponding to the region of influence of the source group of the subdomain.
        """

    @abstractmethod
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

    @abstractmethod
    def resize_local_field(self, local_field: np.ndarray) -> None:
        """Resize a local field to match the size of the local grid.

        Parameters
        ----------
        local_field : The local field to resize (IO parameter).
        """
