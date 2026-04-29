from __future__ import annotations
from abc import abstractmethod
import numpy as np
from pyc2ray.domain.sources import SourceGroup

# ==========================================================================
# This file provides an interface for the Grid class, which is used to 
# represent the full grid of the simulation and the sub-grids corresponding 
# to the subdomains. The idea is to have a common interface for different 
# grid implementations, such as regular grids, octrees, etc. The Grid class
# should provide functionalities for handling the grid data, particularly 
# the mapping between physical coordinates and grid indexes and between 
# local subdomain grid indexes and global grid indexes.
# ==========================================================================

class Grid:
    """Base class for grid representation, providing a common interface for different 
       grid implementations.
        
    Attributes
    ----------
    is_periodic_mode_active : bool
        Flag indicating whether periodic boundary conditions are active in the grid.
    """
    def __init__(self, is_periodic_mode_active: bool = False) -> None:
        self.is_periodic_mode_active = is_periodic_mode_active

    @abstractmethod
    def get_domain_min(self) -> np.ndarray:
        """Get the space coordinates of the minimum corner of the domain.

        Returns
        -------
        np.ndarray
            The minimum corner of the domain (shape `(3,)`).
        """
        raise NotImplementedError("Call to get_domain_min abstract method.")

    @abstractmethod
    def get_domain_max(self) -> np.ndarray:
        """Get the space coordinates of the maximum corner of the domain.

        Returns
        -------
        np.ndarray
            The maximum corner of the domain (shape `(3,)`).
        """
        raise NotImplementedError("Call to get_domain_max abstract method.")

    @abstractmethod
    def global_to_local_map(self, global_field: np.ndarray, local_field: np.ndarray) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.
        
        Parameters
        ----------
        global_field : np.ndarray
            The field defined on the global grid to map.
        local_field : np.ndarray
            The field defined on the local grid initialized with the corresponding values from the global grid. 
            This is an I/O parameter.
        """
        raise NotImplementedError("Call to global_to_local_map abstract method.")


    @abstractmethod
    def local_to_global_map(self, local_field: np.ndarray, global_field: np.ndarray) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid.
        
        Parameters
        ----------
        local_field : np.ndarray
            The field defined on the local grid to map.
        global_field : np.ndarray
            The field defined on the global grid to update with the local field values.
            This is an I/O parameter, which is updated in place with the values of the local field 
            corresponding to the grid elements included in the current grid.
        """
        raise NotImplementedError("Call to local_to_global_map abstract method.")
    
    @abstractmethod
    def get_local_grid(self, source_group: SourceGroup) -> Grid:
        """Get the local grid corresponding to the region of influence of the source group of the subdomain.
        
        Parameters
        ----------
        source_group : SourceGroup
            The group of sources for which to get the local grid.

        Returns
        -------
        Grid
            The local grid corresponding to the region of influence of the source group of the subdomain.
        """
        raise NotImplementedError("Call to get_local_grid abstract method.")
    
    @abstractmethod
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
        raise NotImplementedError("Call to find_num_cells_in_box abstract method.")