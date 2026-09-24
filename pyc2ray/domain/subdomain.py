"""
This file contains the implementation of the Subdomain class, representing a
rectangular box in the simulation domain, and the sources that belong to it.
Besides the data, it also contains functionalities related to the subdomain,
such as handling of subdomain to main domain coordinate transformations.
"""

import numpy as np

from pyc2ray.domain.grid import Grid
from pyc2ray.domain.sources import SourceGroup


class Subdomain:
    """Subdomain representation class, implementing the representation
    of a domain subvolume, namely a subset of main grid and the sources
    which have influence on it.

    Attributes
    ----------
    source_group : SourceGroup
        The group of sources that influence this subvolume.
    local_grid : Grid
        The local grid corresponding to the region of influence of the source group.
    """

    def __init__(self, source_group: SourceGroup, local_grid: Grid) -> None:
        self.source_group = source_group
        self.local_grid = local_grid
        # Lazily built views of the source group, reused for the lifetime of this
        # subdomain: the raytracing loop asks for them once per convergence iteration,
        # while they only change when the decomposition is rebuilt.
        #
        # WARNING: this is correct only because source_group and local_grid are treated
        # as immutable once the subdomain exists -- a rebuild constructs new Subdomain
        # objects rather than updating them in place (see
        # DomainDecompositionHandler._run_decomposition). Nothing enforces that, and
        # mutating a group's source list, or the returned arrays, would make these stale
        # silently: the sources would then be raytraced at the wrong cells with no error.
        # TODO: make this safe for example by returning read-only views.
        self._local_positions: np.ndarray | None = None
        self._local_strengths: np.ndarray | None = None

    def global_to_local_map(
        self, global_field: np.ndarray, local_field: np.ndarray
    ) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        global_field : The field defined on the global grid to map.
        local_field : The field defined on the local grid, initialized with the corresponding
        values from the global grid. This is an I/O parameter.
        """
        self.local_grid.global_to_local_map(global_field, local_field)

    def local_to_global_map(
        self,
        local_field: np.ndarray,
        global_field: np.ndarray,
        add: bool = False,
    ) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid
        and update the global field by adding the local field values.

        It is assumed that the size of the global grid is the one corresponding to the global_field.

        Parameters
        ----------
        local_field : The field defined on the local grid to map.
        global_field : The field defined on the global grid to update with the local field values.
        This is an I/O parameter.
        add : If True, the local field values are added to the global field values. If False, they
        overwrite the corresponding global field values.
        """
        self.local_grid.local_to_global_map(local_field, global_field, add)

    # TODO: this function implicitly assumes that the local grid is a subset of the global grid,
    # which is the case for regular grids but may not be the case for more general grid types.
    # We may need to rethink this interface if we want to support more general grid types in the future.
    def get_local_sources_positions(self) -> np.ndarray:
        """Get the positions of this subvolume's sources as indices in its local grid.

        Returns
        -------
        The positions (indexes) of the sources, referred to the local grid,
        with shape (num_sources, 3).
        """
        if self._local_positions is None:
            sources = self.source_group.sources
            if len(sources) == 0:
                self._local_positions = np.empty((0, 3), dtype=int)
            else:
                positions = np.stack([s.pos for s in sources])
                self._local_positions = self.local_grid.global_to_local_position_map(
                    positions
                )

        return self._local_positions

    def get_local_sources_strengths(self) -> np.ndarray:
        """Get the strengths of this subvolume's sources.

        Returns
        -------
        The strengths of the sources of this subvolume.
        """
        if self._local_strengths is None:
            self._local_strengths = np.fromiter(
                (s.strength for s in self.source_group.sources),
                dtype=np.float64,
                count=len(self.source_group.sources),
            )

        return self._local_strengths

    def resize_local_field(self, local_field: np.ndarray) -> None:
        """Resize a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        local_field : The field defined on the global grid to resize. (This is an I/O parameter.)
        """
        self.local_grid.resize_local_field(local_field)
