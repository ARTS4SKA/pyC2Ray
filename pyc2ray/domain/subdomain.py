from mpi4py import MPI
import numpy as np
from typing import List, Tuple
from pyc2ray.domain.domain_decomposition_utils import log_domain_decomposition_assignments_new
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.morton_grouping import MortonSourceGrouping, MortonGroupingParams
from pyc2ray.domain.source_grouping import GroupingParams
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import get_domain_logger

logger = get_domain_logger(__name__)

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
    source_group : SourceGroup
        Group of sources that have influence on this subdomain.
    local_grid : Grid
        Local sub-grid corresponding to the region of influence of the source 
        group of the subdomain.
    """
    
    def __init__(self, comm: MPI.Comm) -> None:
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.global_grid = None
        self.source_group = None
        self.local_grid = None

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

    def _assign_groups_to_ranks(self, groups: List[SourceGroup]) -> Tuple[List[SourceGroup | None], List[float]]:
        """
        Groups to ranks assignement according to cost.

        Parameters
        ----------
        groups : List[Group]
            List of groups to assign.

        Returns
        -------

        Tuple[List[Group], List[float]]
            A tuple of (rank_groups, rank_costs), where rank_groups is a lists of 
            groups assigned to each rank, and rank_costs is the total cost for each rank.
        """
        rank_groups: List[SourceGroup | None] = [None for _ in range(self.comm.Get_size())]
        rank_costs = [0.0 for _ in range(self.comm.Get_size())]

        # TODO: this is a basic assignment. More sophisticated algorithms 
        # should be used for better load balancing.
        for g in sorted(groups, key=lambda x: x.cost, reverse=True):
            r = int(np.argmin(rank_costs))
            rank_groups[r] = g
            rank_costs[r] = g.cost

        return rank_groups, rank_costs

    def get_local_grid(self) -> Grid:
        """ Get the local grid corresponding to group of sources assigned to this rank.

        Returns
        -------
        Grid
            The local grid corresponding to the assigned group of sources
        """
        if self.local_grid is None:
            raise ValueError("No source group assigned to rank, cannot get local grid.")

        return self.local_grid

    def get_source_group(self) -> SourceGroup:
        """ Get the group of sources assigned to this rank.

        Returns
        -------
        SourceGroup
            The group of sources assigned to this rank.
        """
        if self.source_group is None:
            raise ValueError("No source group assigned to rank, cannot get source group.")

        return self.source_group

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

            print("Built {} groups with {} grouping.".format(len(groups), grouping_algorithm))

            # TODO: remove this check and implement the correct procedures for zero or multiple groups per rank, which are currently not supported.
            if len(groups) != self.comm.Get_size():
                raise NotImplementedError(f"Error: Number of groups ({len(groups)}) is different from the number of ranks ({self.comm.Get_size()}). This is not supported yet.")

            # Assign the groups to the ranks
            ranks_groups, ranks_costs = self._assign_groups_to_ranks(groups)
            
            # Log the assignments for debugging purposes
            log_domain_decomposition_assignments_new(ranks_groups, ranks_costs)
        else:
            ranks_groups = None
        
        # Scatter the groups to the ranks
        self.source_group = self.comm.scatter(ranks_groups, root=0)

        # TODO: remove this check and implementend the correct procedures for zero
        # or multiple groups per rank, which are currently not supported.
        if self.source_group is None:
            raise NotImplementedError(f"Error in rank {self.rank}: No group assigned to rank, which is not supported yet.")

        # Retrieve the local grid corresponding to the assigned group of sources
        self.local_grid = self.global_grid.get_local_grid(self.source_group)

