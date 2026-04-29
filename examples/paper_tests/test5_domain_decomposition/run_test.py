from __future__ import annotations

import mpi4py
mpi4py.rc.finalize = False
from mpi4py import MPI  # noqa: E402
import numpy as np

from pyc2ray.domain.morton_grouping import MortonGroupingParams
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source
from pyc2ray.domain.subdomain import Subdomain

# def generate_sources(num_cluster_sources: int, cluster_center: np.ndarray, cluster_width: float,
#                      num_sparse_sources: int, source_strength: float, r_max_lls: float, boxsize: float) -> List[Source]:
#     """Generate a list of sources for the toy example, with a clustered distribution and a sparse distribution.
#     """
#     sources = []
#     # Clustered sources
#     for i in range(num_cluster_sources):
#         pos = cluster_center + np.random.uniform(-cluster_width/2, cluster_width/2, size=3)
#         sources.append(Source(position=pos, luminosity=source_strength))
#     # Sparse sources
#     for i in range(num_sparse_sources):
#         pos = np.random.uniform(0, boxsize, size=3)
#         sources.append(Source(position=pos, luminosity=source_strength))
#     return sources

def main():
    """Run a complete toy example: source generation, grouping, and plotting."""

    # Create the global grid
    global_grid = RegularGrid(cell_size=1, num_cells=100)

    # Create subdomain handler
    subdomain = Subdomain(MPI.COMM_WORLD)

    # Create source list
    sources = [
        Source(id=0, pos=np.array([0.5, 0.5, 0.5]), strength=1.0, radius=1.0),
        Source(id=1, pos=np.array([1.5, 1.5, 1.5]), strength=1.0, radius=1.0),
        Source(id=2, pos=np.array([0.5, 1.5, 0.5]), strength=1.0, radius=1.0),
        Source(id=3, pos=np.array([0.5, 0.5, 1.5]), strength=1.0, radius=1.0),
        Source(id=4, pos=np.array([1.0, 1.0, 1.0]), strength=1.0, radius=1.0),
        Source(id=5, pos=np.array([60.0, 60.0, 60.0]), strength=1.0, radius=1.0),
        Source(id=6, pos=np.array([70.0, 70.0, 70.0]), strength=1.0, radius=1.0),
    ]

    # Run the domain decomposition
    subdomain.run_decomposition(global_grid, sources, grouping_algorithm="morton", 
                                grouping_params = MortonGroupingParams(max_num_sources_per_group=2, 
                                                                       max_cost_per_group=2.0, 
                                                                       morton_bits=5))
    MPI.Finalize()

if __name__ == "__main__":
    main()
