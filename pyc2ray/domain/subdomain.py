
from mpi4py import MPI
import numpy as np

from typing import List, Tuple

from pyc2ray.domain.grid import Grid
from pyc2ray.domain.morton_grouping import MortonSourceGrouping, MortonGroupingParams
from pyc2ray.domain.source_grouping import GroupingParams
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import log_domain_decomposition_assignments

# TODO: split the functionalities of this class between a Subdomain class, 
# which contains the data of a specific subdomain and implements its functionalities
# (mainly handling of subdomain to main domain coordinate transformations, and 
# communication between subdomains), and a DomainDecompositionHandler class, which implements 
# the domain decomposition algorithm and the assignment of groups to ranks.
# The DomainDecompositionHandler could also be useful in case of dynamic load balancing 
# where the domain decomposition and assignment to ranks would need to be redone during 
# the simulation, while the Subdomain class would still be responsible for handling the 
# data and functionalities of the subdomains, independently of how they are assigned to ranks.
# Moreover, this separation could also be useful in case if it's needed to assign more than one 
# group to a rank.
  
# ======================================================================
# This file contains the implementetion of the Subdomain class, which 
# contains the data of a specific subdomain and implements its 
# functionalities. A subdomain is a rectangular box in the simulation 
# domain, and the sources that belong to it. Besides the data, it also 
# contains functionalities related to the subdomain, such as handling of 
# subdomain to main domain coordinate transformations, and communication 
# between subdomains.
# ======================================================================

class Subdomain:

    """ Subdomain representation class, implementing the representation
    of a domain subvolume, namely a subset of main grid and the sources
    which have influence on it during the ionization process. 

    Attributes
    ----------
    rank : int
        MPI rank of the subdomain.
    comm : MPI.Comm
        MPI communicator for the subdomain
    grid : Grid
        Grid object representing the subset of the main grid that belongs 
        to this subdomain.
    source_groups : List[SourceGroup]
        List of source groups that have influence on this subdomain.
    local_grids : List[Grid]
        List of local sub-grids corresponding to the regions of influence of the source
        groups of the subdomain.
    """
    
    def __init__(self, comm: MPI.Comm) -> None:
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.global_grid = None
        self.source_groups : List[SourceGroup] | None = None
        self.local_grids : List[Grid] | None = None
        self.cost = 0.0

    def _build_groups(self, global_grid: Grid, sources: List[Source], grouping_algorithm: str = "morton", 
                      grouping_params: GroupingParams = MortonGroupingParams()) -> List[SourceGroup]:
        """ Build the groups of sources to be assigned to the ranks.

        Parameters
        ----------
        global_grid : Grid
            The full grid of the simulation.
        sources : List[Source]
            The full list of sources in the simulation.
        grouping_algorithm : str
            The algorithm to use for source grouping/domain decomposition.
        grouping_params : GroupingParams
            The parameters for the source grouping/domain decomposition algorithm.

        Returns
        -------
        List[SourceGroup]
            The list of source groups to be assigned to the ranks.
        """
        if grouping_algorithm == "morton":
            return MortonSourceGrouping().build_groups(sources, global_grid, grouping_params)
        else:
            raise NotImplementedError(f"Grouping algorithm {grouping_algorithm} not implemented yet.")

    def _assign_groups_to_ranks(self, groups: List[SourceGroup]) -> Tuple[List[List[SourceGroup]], List[float]]:
        """
        Groups to ranks assignement according to cost.

        Parameters
        ----------
        groups : List[Group]
            List of groups to assign.

        Returns
        -------

        Tuple[List[List[SourceGroup]], List[float]]
            A tuple of (rank_groups, rank_costs), where rank_groups is a list of lists of
            groups assigned to each rank, and rank_costs is the total cost for each rank.
        """
        rank_groups: List[List[SourceGroup]] = [[] for _ in range(self.comm.Get_size())]
        rank_costs = [0.0 for _ in range(self.comm.Get_size())]

        # TODO: this is a basic assignment. More sophisticated algorithms 
        # should be used for better load balancing.
        for g in sorted(groups, key=lambda x: x.cost, reverse=True):
            r = int(np.argmin(rank_costs))
            rank_groups[r].append(g)
            rank_costs[r] += g.cost

        return rank_groups, rank_costs

    def get_local_grids(self) -> List[Grid]:
        """ Get the local grids corresponding to the groups assigned to this rank.

        Returns
        -------
        List[Grid]
            The local grids corresponding to the assigned groups of sources.
        """
        if self.local_grids is None:
            raise ValueError("No source groups assigned to rank, cannot get local grids.")

        return self.local_grids

    def run_decomposition(self, global_grid: Grid, sources: List[Source], grouping_algorithm: str = "morton", 
                          grouping_params: GroupingParams = MortonGroupingParams()) -> None:
        """ Run the domain decomposition, which consists in building the groups 
        of sources and the corresponding grid subvolumes that belong together, 
        and assigning them to the ranks.

        Parameters
        ----------
        global_grid : Grid
            The full grid of the simulation.
        sources : List[Source]
            The full list of sources in the simulation.
        grouping_algorithm : str
            The algorithm to use for source grouping/domain decomposition.
        grouping_params : GroupingParams
            The parameters for the source grouping/domain decomposition algorithm.
        """
        self.global_grid = global_grid
        if self.rank == 0:
            # Build the groups of sources to be assigned to the ranks
            groups = self._build_groups(global_grid, sources, grouping_algorithm=grouping_algorithm, 
                                        grouping_params=grouping_params)

            # Assign the groups to the ranks
            ranks_groups, ranks_costs = self._assign_groups_to_ranks(groups)
            
            # Log the assignments for debugging purposes
            log_domain_decomposition_assignments(ranks_groups, ranks_costs)
        else:
            ranks_groups = None
        
        # Scatter the groups to the ranks
        self.source_groups = self.comm.scatter(ranks_groups, root=0)

        # Retrieve the local grid corresponding to the assigned group of sources
        if self.source_groups is not None:
            self.local_grids = [self.global_grid.get_local_grid(group) for group in self.source_groups]
        else:
            self.local_grids = []

    def global_to_local_map(self, subgrid_index: int, global_field: np.ndarray, local_field: np.ndarray) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        subgrid_index : int
            The index of the subgrid to map.
        global_field : np.ndarray
            The field defined on the global grid to map.
        local_field : np.ndarray
            The field defined on the local grid initialized with the corresponding values from the global grid.
            This is an I/O parameter.
        """
        if self.local_grids is None:
            raise ValueError("No source groups assigned to rank, cannot perform global to local mapping.")

        if subgrid_index >= len(self.local_grids):
            raise ValueError(f"Subgrid index {subgrid_index} out of range for local grids assigned to rank.")

        self.local_grids[subgrid_index].global_to_local_map(global_field, local_field)

    def local_to_global_map(self, subgrid_index: int, local_field: np.ndarray, global_field: np.ndarray) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid
        and update the global field by adding the local field values.

        It is assumed that the size of the global grid is the one corresponding to the global_field.

        Parameters
        ----------
        subgrid_index : int
            The index of the subgrid to map.
        local_field : np.ndarray
            The field defined on the local grid to map.
        global_field : np.ndarray
            The field defined on the global grid to update with the local field values.
            This is an I/O parameter, which is updated in place with the values of the local field
            corresponding to the grid elements included in the current grid.
        """
        if self.local_grids is None:
            raise ValueError("No source groups assigned to rank, cannot perform local to global mapping.")

        if subgrid_index >= len(self.local_grids):
            raise ValueError(f"Subgrid index {subgrid_index} out of range for local grids assigned to rank.")

        self.local_grids[subgrid_index].local_to_global_map(local_field, global_field)

    def get_source_group(self, subdomain_index: int) -> SourceGroup:
        """Get the source group corresponding to the given subdomain index.

        Parameters
        ----------
        subdomain_index : int
            The index of the subdomain for which to get the corresponding source group.

        Returns
        -------
        SourceGroup
            The source group corresponding to the given subdomain index.
        """
        if self.source_groups is None:
            raise ValueError("No source groups assigned to rank, cannot get source group.")

        if subdomain_index >= len(self.source_groups):
            raise ValueError(f"Subdomain index {subdomain_index} out of range for source groups assigned to rank.")

        return self.source_groups[subdomain_index]

    def get_source_groups(self) -> List[SourceGroup]:
        """Get the list of source groups assigned to this rank.

        Returns
        -------
        List[SourceGroup]
            The list of source groups assigned to this rank.
        """
        if self.source_groups is None:
            raise ValueError("No source groups assigned to rank, cannot get source groups.")

        return self.source_groups

    # TODO: this function implicitly assumes that the local grid is a subset of the global grid, which is the case for regular grids but may not be the case for more general grid types.
    # We may need to rethink this interface if we want to support more general grid types in the future.
    def get_local_sources_positions(self, subdomain_index: int) -> np.ndarray:
        """Get the positions (indexes) of the sources corresponding to the given subdomain index.

        The indexes returned by this functions refers to the local grid.

        Parameters
        ----------
        subdomain_index : int
            The index of the subdomain for which to get the corresponding source positions.

        Returns
        -------
        np.ndarray
            The positions (indexes) of the sources corresponding to the given subdomain index.
        """
        source_group = self.get_source_group(subdomain_index)
        local_grid = self.get_local_grids()[subdomain_index]
        return np.array([[local_grid.global_to_local_position_map(s.pos)[i] for s in source_group.sources] for i in range(3)])

    def get_num_source_groups(self) -> int:
        """Get the number of source groups assigned to this rank.

        Returns
        -------
        int
            The number of source groups assigned to this rank.
        """
        if self.source_groups is None:
            raise ValueError("No source groups assigned to rank, cannot get number of source groups.")

        return len(self.source_groups)

    def get_local_sources_strengths(self, subdomain_index: int) -> np.ndarray:
        """Get the strengths of the sources corresponding to the given subdomain index.

        Parameters
        ----------
        subdomain_index : int
            The index of the subdomain for which to get the corresponding source strengths.

        Returns
        -------
        np.ndarray
            The strengths of the sources corresponding to the given subdomain index.
        """
        source_group = self.get_source_group(subdomain_index)
        return np.array([s.strength for s in source_group.sources])

    def resize_local_field(self, subdomain_index: int, local_field: np.ndarray) -> None:
        """Resize a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        subdomain_index : int
            The index of the subdomain for which to resize the field.
        local_field : np.ndarray
            The field defined on the global grid to resize.

        Returns
        -------
        np.ndarray
            The resized field defined on the local grid.
        """
        if self.local_grids is None:
            raise ValueError("No source groups assigned to rank, cannot perform field resizing.")

        if subdomain_index >= len(self.local_grids):
            raise ValueError(f"Subdomain index {subdomain_index} out of range for local grids assigned to rank.")

        local_grid = self.local_grids[subdomain_index]
        return local_grid.resize_local_field(local_field)