"""
This file contains the implementation of the Subdomain class, which
contains the data of a specific subdomain and implements its
functionalities. A subdomain is a rectangular box in the simulation
domain, and the sources that belong to it. Besides the data, it also
contains functionalities related to the subdomain, such as handling of
subdomain to main domain coordinate transformations, and communication
between subdomains.
"""

from typing import Protocol, cast

import numpy as np

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.morton_grouping import MortonGroupingParams, MortonSourceGrouping
from pyc2ray.domain.source_grouping import GroupingParams
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import log_domain_decomposition_assignments


class Comm(Protocol):
    """Minimal MPI-like communicator interface required by Subdomain
    testing and Python code type annotations check.
    """

    def Get_rank(self) -> int: ...
    def Get_size(self) -> int: ...
    def scatter(self, data: object, root: int = 0) -> object: ...


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


class Subdomain:
    """Subdomain representation class, implementing the representation
    of a domain subvolume, namely a subset of main grid and the sources
    which have influence on it during the ionization process.

    Attributes
    ----------
    rank : int
        Rank of the subdomain.
    comm : Comm
        Communicator for the subdomain
    grid : Grid
        Grid object representing the subset of the main grid that belongs
        to this subdomain.
    source_groups : list[SourceGroup]
        List of source groups that have influence on this subdomain.
    local_grids : list[Grid]
        List of local sub-grids corresponding to the regions of influence of the source
        groups of the subdomain.
    """

    def __init__(self, comm: Comm) -> None:
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.global_grid: Grid | None = None
        self.source_groups: list[SourceGroup] | None = None
        self.local_grids: list[Grid] | None = None
        self.cost = 0.0

    def _build_groups(
        self,
        global_grid: Grid,
        sources: list[Source],
        cost_model: CostModel,
        grouping_algorithm: str = "morton",
        grouping_params: GroupingParams | None = None,
    ) -> list[SourceGroup]:
        """Build the groups of sources to be assigned to the ranks.

        Parameters
        ----------
        global_grid : The full grid of the simulation.
        sources : The full list of sources in the simulation.
        cost_model : The cost model to use for the evaluation of the cost of processing a group of sources.
        grouping_algorithm : The algorithm to use for source grouping/domain decomposition.
        grouping_params : The parameters for the source grouping/domain decomposition algorithm.

        Returns
        -------
        The list of source groups to be assigned to the ranks.
        """
        if grouping_params is None:
            grouping_params = MortonGroupingParams()

        if grouping_algorithm == "morton":
            return MortonSourceGrouping().build_groups(
                sources, global_grid, grouping_params, cost_model
            )

        raise NotImplementedError(
            f"Grouping algorithm {grouping_algorithm} not implemented yet."
        )

    def _assign_groups_to_ranks(
        self, groups: list[SourceGroup]
    ) -> tuple[list[list[SourceGroup]], list[float]]:
        """
        Groups to ranks assignment according to cost.

        Parameters
        ----------
        groups : List of groups to assign.

        Returns
        -------

        A tuple of (rank_groups, rank_costs), where rank_groups is a list of lists of
        groups assigned to each rank, and rank_costs is the total cost for each rank.
        """
        rank_groups: list[list[SourceGroup]] = [[] for _ in range(self.comm.Get_size())]
        rank_costs = [0.0 for _ in range(self.comm.Get_size())]

        # TODO: this is a basic assignment. More sophisticated algorithms
        # should be used for better load balancing.
        for g in sorted(groups, key=lambda x: x.comp_cost, reverse=True):
            r = int(np.argmin(rank_costs))
            rank_groups[r].append(g)
            rank_costs[r] += g.comp_cost

        return rank_groups, rank_costs

    def get_local_grids(self) -> list[Grid]:
        """Get the local grids corresponding to the groups assigned to this rank.

        Returns
        -------
        The local grids corresponding to the assigned groups of sources.
        """
        if self.local_grids is None:
            raise ValueError(
                "No source groups assigned to rank, cannot get local grids."
            )

        return self.local_grids

    def run_decomposition(
        self,
        global_grid: Grid,
        sources: list[Source],
        cost_model: CostModel,
        grouping_algorithm: str = "morton",
        grouping_params: GroupingParams | None = None,
    ) -> None:
        """Run the domain decomposition, which consists in building the groups
        of sources and the corresponding grid subvolumes that belong together,
        and assigning them to the ranks.

        Parameters
        ----------
        global_grid : The full grid of the simulation.
        sources : The full list of sources in the simulation.
        cost_model : The cost model to use for the evaluation of the cost of processing a group of sources.
        grouping_algorithm : The algorithm to use for source grouping/domain decomposition.
        grouping_params : The parameters for the source grouping/domain decomposition algorithm.
        """
        self.global_grid = global_grid
        if self.rank == 0:
            # Build the groups of sources to be assigned to the ranks
            groups = self._build_groups(
                global_grid,
                sources,
                cost_model=cost_model,
                grouping_algorithm=grouping_algorithm,
                grouping_params=grouping_params,
            )

            # Assign the groups to the ranks
            ranks_groups, ranks_costs = self._assign_groups_to_ranks(groups)

            # Log the assignments for debugging purposes
            log_domain_decomposition_assignments(ranks_groups, ranks_costs)
        else:
            ranks_groups = None

        # Scatter the groups to the ranks
        self.source_groups = cast(
            list[SourceGroup] | None,
            self.comm.scatter(ranks_groups, root=0),
        )

        # Retrieve the local grid corresponding to the assigned group of sources
        if self.source_groups is not None:
            if self.global_grid is None:
                raise ValueError(
                    "Global grid is not initialized, cannot build local grids."
                )
            self.local_grids = [
                self.global_grid.get_local_grid(group) for group in self.source_groups
            ]
        else:
            self.local_grids = []

        # Update the cost of the subdomain with the cost of the assigned groups
        self.cost = (
            sum(g.comp_cost for g in self.source_groups)
            if self.source_groups is not None
            else 0.0
        )

    def global_to_local_map(
        self, subgrid_index: int, global_field: np.ndarray, local_field: np.ndarray
    ) -> None:
        """Map a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        subgrid_index : The index of the subgrid to map.
        global_field : The field defined on the global grid to map.
        local_field : The field defined on the local grid initialized with the corresponding values from
        the global grid.This is an I/O parameter.
        """
        if self.local_grids is None:
            raise ValueError(
                "No source groups assigned to rank, cannot perform global to local mapping."
            )

        if subgrid_index >= len(self.local_grids):
            raise ValueError(
                f"Subgrid index {subgrid_index} out of range for local grids assigned to rank."
            )

        self.local_grids[subgrid_index].global_to_local_map(global_field, local_field)

    def local_to_global_map(
        self,
        subgrid_index: int,
        local_field: np.ndarray,
        global_field: np.ndarray,
        add: bool = False,
    ) -> None:
        """Map a field defined on the local grid to the corresponding field on the global grid
        and update the global field by adding the local field values.

        It is assumed that the size of the global grid is the one corresponding to the global_field.

        Parameters
        ----------
        subgrid_index : The index of the subgrid to map.
        local_field : The field defined on the local grid to map.
        global_field : The field defined on the global grid to update with the local field values.
        This is an I/O parameter, which is updated in place with the values of the local field
        corresponding to the grid elements included in the current grid.
        add : If True, the local field values are added to the global field values. If False, the local
        field values are set to the global field values.
        """
        if self.local_grids is None:
            raise ValueError(
                "No source groups assigned to rank, cannot perform local to global mapping."
            )

        if subgrid_index >= len(self.local_grids):
            raise ValueError(
                f"Subgrid index {subgrid_index} out of range for local grids assigned to rank."
            )

        self.local_grids[subgrid_index].local_to_global_map(
            local_field, global_field, add
        )

    # TODO: it would be more pythonic to call this method __getitem__
    def get_source_group(self, subdomain_index: int) -> SourceGroup:
        """Get the source group corresponding to the given subdomain index.

        Parameters
        ----------
        subdomain_index : The index of the subdomain for which to get the corresponding source group.

        Returns
        -------
        The source group corresponding to the given subdomain index.
        """
        if self.source_groups is None:
            raise ValueError(
                "No source groups assigned to rank, cannot get source group."
            )

        if subdomain_index >= len(self.source_groups):
            raise ValueError(
                f"Subdomain index {subdomain_index} out of range for source groups assigned to rank."
            )

        return self.source_groups[subdomain_index]

    def get_source_groups(self) -> list[SourceGroup]:
        """Get the list of source groups assigned to this rank.

        Returns
        -------
        The list of source groups assigned to this rank.
        """
        if self.source_groups is None:
            raise ValueError(
                "No source groups assigned to rank, cannot get source groups."
            )

        return self.source_groups

    # TODO: this function implicitly assumes that the local grid is a subset of the global grid, which is the case for regular grids but may not be the case for more general grid types.
    # We may need to rethink this interface if we want to support more general grid types in the future.
    def get_local_sources_positions(self, subdomain_index: int) -> np.ndarray:
        """Get the positions (indexes) of the sources corresponding to the given subdomain index.

        The indexes returned by this functions refers to the local grid.

        Parameters
        ----------
        subdomain_index : The index of the subdomain for which to get the corresponding source positions.

        Returns
        -------
        The positions (indexes) of the sources corresponding to the given subdomain index.
        """
        source_group = self.get_source_group(subdomain_index)
        local_grid = self.get_local_grids()[subdomain_index]
        return np.array(
            [
                [
                    local_grid.global_to_local_position_map(s.pos)[i]
                    for s in source_group.sources
                ]
                for i in range(3)
            ]
        )

    def get_num_source_groups(self) -> int:
        """Get the number of source groups assigned to this rank.

        Returns
        -------
        The number of source groups assigned to this rank.
        """
        if self.source_groups is None:
            raise ValueError(
                "No source groups assigned to rank, cannot get number of source groups."
            )

        return len(self.source_groups)

    def get_local_sources_strengths(self, subdomain_index: int) -> np.ndarray:
        """Get the strengths of the sources corresponding to the given subdomain index.

        Parameters
        ----------
        subdomain_index : The index of the subdomain for which to get the corresponding source strengths.

        Returns
        -------
        The strengths of the sources corresponding to the given subdomain index.
        """
        source_group = self.get_source_group(subdomain_index)
        return np.array([s.strength for s in source_group.sources])

    def resize_local_field(self, subdomain_index: int, local_field: np.ndarray) -> None:
        """Resize a field defined on the global grid to the corresponding field on the local grid.

        Parameters
        ----------
        subdomain_index : The index of the subdomain for which to resize the field.
        local_field : The field defined on the global grid to resize.

        Returns
        -------
        The resized field defined on the local grid.
        """
        if self.local_grids is None:
            raise ValueError(
                "No source groups assigned to rank, cannot perform field resizing."
            )

        if subdomain_index >= len(self.local_grids):
            raise ValueError(
                f"Subdomain index {subdomain_index} out of range for local grids assigned to rank."
            )

        self.local_grids[subdomain_index].resize_local_field(local_field)
