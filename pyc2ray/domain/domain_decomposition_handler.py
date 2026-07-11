"""
This file contains the implementation of the DomainDecompositionHandler class,
which runs the source grouping / domain decomposition, assigns the resulting
source groups to the MPI ranks, and holds the Subdomains assigned to the current rank.
Moreover, it handles the the decomposition rebuild vs reuse logic.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np

from pyc2ray.domain.cost_model import CostModel, pyC2RayCostModel
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.morton_grouping import MortonGroupingParams, MortonSourceGrouping
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.source_grouping import GroupingParams
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain
from pyc2ray.domain.utils import log_domain_decomposition_assignments
from pyc2ray.parameters import DomainDecompositionParameters

logger = logging.getLogger(__name__)


class Comm(Protocol):
    """Minimal MPI-like communicator interface required by DomainDecompositionHandler
    and by the Python code type annotations check.
    """

    def Get_rank(self) -> int: ...
    def Get_size(self) -> int: ...
    def scatter(self, sendobj: Sequence[Any] | None, root: int = 0) -> Any: ...


# TODO: split this class into a generic base and a pyC2Ray-specific subclass. Most of the
# class is grid-agnostic (grouping, rank assignment, scatter, caching, get_subdomains), but
# update_decomposition and _build_inputs are coupled to pyC2Ray: their input types are raw
# pyC2Ray simulation data and _build_inputs constructs RegularGrid / pyC2RayCostModel /
# MortonGroupingParams. The generic machinery (including a decompose(grid, sources, cost_model,
# ..., key) entry point doing the caching + _run_decomposition) would stay in the base, while a
# pyC2RayDomainDecompositionHandler subclass would provide update_decomposition (its raw-data
# signature + key formula) and _build_inputs. This is the natural seam for supporting other grid
# types (e.g. AMR): a different subclass would only reimplement those two methods. Alternatively
# the Domain module could completely rely on numpy arrays to avoid translation overhead.
class DomainDecompositionHandler:
    """Source grouping/domain decomposition handling class, implementing the logic for
    grouping sources and decomposing the simulation domain. Owns the rebuild-vs-reuse
    decision depending on the external inputs, and holds the subdomains assigned to the
    current rank.

    Attributes
    ----------
    rank : int
        Rank of the current process.
    comm : Comm
        MPI Communicator
    global_grid : Grid
        The full grid of the simulation.
    subdomains : list[Subdomain]
        The subvolumes (one per assigned source group) assigned to the current rank.
    cost : float
        Total computational cost of the subdomains assigned to the current rank.
    decomposition_key : tuple | None
        Key identifying the grouping inputs the current decomposition was built with,
        used by update_decomposition to decide when a rebuild is necessary.
    """

    def __init__(self, comm: Comm) -> None:
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.global_grid: Grid | None = None
        self.subdomains: list[Subdomain] = []
        self.cost = 0.0
        # Key identifying the grouping inputs the current decomposition was built with,
        # used by update_decomposition to decide when a rebuild is necessary. Internal only.
        # TODO: save memory by using a hash instead of the full bytes representation of the arrays.
        self._decomposition_key: tuple | None = None

    def update_decomposition(
        self,
        cell_size: float,
        src_pos: np.ndarray,
        src_flux: np.ndarray,
        N: int,
        R_max_LLS: float,
        src_batch_size: int,
        num_tau: int,
        is_domain_periodic: bool,
        domain_decomposition_params: DomainDecompositionParameters,
    ) -> None:
        """Update the domain decomposition according to the specified parameters if needed.
        Rebuild the decomposition if it is the first time, or if the inputs have changed with respect to the last build.
        Otherwise, reuse the cached decomposition.

        Parameters
        ----------
        cell_size : Size of the grid cells (assuming cubic cells).
        src_pos : Array containing the 3D grid position of each source, in C-style indexing (a source in a cell at the origin
        has position (0, 0, 0)).
        src_flux : Array containing the total ionizing flux of each source.
        N : Number of cells per side of the (cubic) global grid.
        R_max_LLS : Mean free path of photons, in cell units.
        src_batch_size : Number of sources processed per batch, used by the cost model.
        num_tau : Size of the photoionization table, used by the cost model.
        is_domain_periodic : Whether the global domain uses periodic boundary conditions.
        domain_decomposition_params : Parameters for the domain decomposition and source grouping algorithm.
        """
        src_pos = src_pos.astype(np.int32)
        src_flux = src_flux.astype(np.float64)

        # Key identifying the grouping inputs. cell_size and N are deliberately excluded: the
        # grouping is invariant under a uniform cell-size scaling, so a cosmological cell-size
        # change alone does not require a rebuild.
        decomposition_key = (src_pos.tobytes(), src_flux.tobytes(), float(R_max_LLS))
        if decomposition_key == self._decomposition_key:
            logger.info("Reusing cached source grouping/domain decomposition.")
            return

        global_grid, sources, cost_model, grouping_params = self._build_inputs(
            cell_size,
            src_pos,
            src_flux,
            N,
            R_max_LLS,
            src_batch_size,
            num_tau,
            is_domain_periodic,
            domain_decomposition_params,
        )

        # TODO: grouping_algorithm could be retrieved from grouping_params instead of passing it separately
        self._run_decomposition(
            global_grid,
            sources,
            cost_model=cost_model,
            grouping_algorithm=domain_decomposition_params.grouping_algorithm,
            grouping_params=grouping_params,
        )
        self._decomposition_key = decomposition_key
        logger.info("Built source grouping/domain decomposition.")

    def _build_inputs(
        self,
        cell_size: float,
        src_pos: np.ndarray,
        src_flux: np.ndarray,
        N: int,
        R_max_LLS: float,
        src_batch_size: int,
        num_tau: int,
        is_domain_periodic: bool,
        domain_decomposition_params: DomainDecompositionParameters,
    ) -> tuple[Grid, list[Source], CostModel, GroupingParams | None]:
        """Build the objects the grouping operates on from the raw simulation data.

        This is the regular-grid specialization of the decomposition: a future adaptive
        mesh refinement handler would override only this step, leaving the grouping,
        assignment and caching untouched.

        Returns
        -------
        The global grid, the source list, the cost model and the grouping parameters.
        """
        num_src, *_ = src_flux.shape

        # Create source grouping grid and source ojects
        global_grid = RegularGrid(
            cell_size=cell_size, num_cells=N, is_periodic_mode_active=is_domain_periodic
        )
        sources = [
            Source(
                id=i,
                pos=(np.array(src_pos[:, i], dtype=float) + 0.5) * cell_size,
                strength=src_flux[i],
                radius=R_max_LLS * cell_size,
            )
            for i in range(num_src)
        ]

        # Create grouping parameters
        # TODO: remove the if and create a factory function in the domain module
        grouping_params: GroupingParams | None = None
        if domain_decomposition_params.grouping_algorithm == "morton":
            grouping_params = MortonGroupingParams(
                max_num_sources_per_group=domain_decomposition_params.max_num_sources_per_group,
                morton_bits=domain_decomposition_params.morton_bits,
            )

        # Set cost model
        cost_model = pyC2RayCostModel(
            max_memory_cost_per_group=domain_decomposition_params.max_memory_cost_per_group,
            source_batch_size=src_batch_size,
            is_periodic_mode_active=is_domain_periodic,
            photo_ion_table_size=num_tau,
        )

        return global_grid, sources, cost_model, grouping_params

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

    def _run_decomposition(
        self,
        global_grid: Grid,
        sources: list[Source],
        cost_model: CostModel,
        grouping_algorithm: str = "morton",
        grouping_params: GroupingParams | None = None,
    ) -> None:
        """Run the domain decomposition: build the source groups, assign them to the
        ranks and store, for the current rank, the corresponding Subdomains.

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
        local_groups = cast(
            list[SourceGroup] | None,
            self.comm.scatter(ranks_groups, root=0),
        )
        if local_groups is None:
            local_groups = []

        # Build the subdomains (source group + local grid) assigned to this rank.
        self.subdomains = [
            Subdomain(group, global_grid.get_local_grid(group))
            for group in local_groups
        ]

        # Update the cost of this rank with the cost of the assigned groups.
        self.cost = sum(sd.source_group.comp_cost for sd in self.subdomains)

    def get_subdomains(self) -> list[Subdomain]:
        """Get the subdomains assigned to the current rank.

        Returns
        -------
        The list of subdomains (source group + local grid) assigned to the current rank.
        """
        return self.subdomains

    def get_num_subdomains(self) -> int:
        """Get the number of subdomains assigned to the current rank.

        Returns
        -------
        The number of subdomains assigned to the current rank.
        """
        return len(self.subdomains)
