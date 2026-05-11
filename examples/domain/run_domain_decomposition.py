from __future__ import annotations

import mpi4py
mpi4py.rc.finalize = False
from mpi4py import MPI
import numpy as np

from pyc2ray.domain.morton_grouping import MortonGroupingParams
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain
from pyc2ray.domain.utils import get_domain_logger

logger = get_domain_logger(__name__)

def main():
    """Run the domain decomposition step and plot the resulting source groups and local grids."""

    # Create the global grid
    global_grid = RegularGrid(cell_size=1, num_cells=100)

    # Create subdomain handler
    subdomain = Subdomain(MPI.COMM_WORLD)

    # Create source list
    sources = [
        Source(id=0, pos=np.array([10.5, 10.5, 10.5]), strength=1.0, radius=1.0),
        Source(id=1, pos=np.array([11.5, 11.5, 11.5]), strength=1.0, radius=1.0),
        Source(id=2, pos=np.array([10.5, 11.5, 10.5]), strength=1.0, radius=1.0),
        Source(id=3, pos=np.array([10.5, 10.5, 11.5]), strength=1.0, radius=1.0),
        Source(id=4, pos=np.array([10.0, 10.0, 10.0]), strength=1.0, radius=1.0),
        Source(id=5, pos=np.array([60.0, 60.0, 60.0]), strength=1.0, radius=1.0),
        Source(id=6, pos=np.array([70.0, 70.0, 70.0]), strength=1.0, radius=1.0),
    ]

    if subdomain.rank == 0:
        logger.info("Running domain decompositon with {} ranks.".format(subdomain.comm.Get_size()))

    # Run the domain decomposition
    subdomain.run_decomposition(global_grid, sources, grouping_algorithm="morton", 
                                grouping_params = MortonGroupingParams(max_num_sources_per_group=2,
                                                                       max_cost_per_group=1.0,
                                                                       morton_bits=10))

    # Collect source groups from all ranks
    source_groups_by_rank: list[list[SourceGroup]] | None = subdomain.comm.gather(
        subdomain.get_source_groups(), root=0
    )

    # Collect local grids from all ranks
    local_grids_by_rank: list[list[RegularGrid]] | None = subdomain.comm.gather(
        subdomain.get_local_grids(), root=0
    )

    # # Plot domain decomposition
    # if subdomain.rank == 0:
    #     assert source_groups_by_rank is not None
    #     assert local_grids_by_rank is not None
    #     source_groups = [group for rank_groups in source_groups_by_rank for group in rank_groups]
    #     local_grids = [grid for rank_grids in local_grids_by_rank for grid in rank_grids]
    #     plot_domain_decomposition(global_grid, source_groups, local_grids)


    MPI.Finalize()

if __name__ == "__main__":
    main()
