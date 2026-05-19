from __future__ import annotations

import argparse

import mpi4py
mpi4py.rc.finalize = False
from mpi4py import MPI
import numpy as np

from pyc2ray.domain.cost_model import pyC2RayCostModel
from pyc2ray.domain.morton_grouping import MortonGroupingParams
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain
from pyc2ray.domain.utils import get_domain_logger
from pyc2ray.visualization.domain_decomposition import export_domain_decomposition_npz, plot_domain_decomposition

logger = get_domain_logger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small domain decomposition example.")
    parser.add_argument("--skip-png", action="store_true", help="Skip PNG plotting to reduce runtime.")
    parser.add_argument("--skip-npz", action="store_true", help="Skip writing NPZ decomposition dump.")
    return parser.parse_args()

def main():
    """Run the domain decomposition step and plot the resulting source groups and local grids."""

    args = _parse_args()

    # Create the global grid
    global_grid = RegularGrid(cell_size=1.0, num_cells=100)

    # Create subdomain handler
    subdomain = Subdomain(MPI.COMM_WORLD)

    # Create source list: 50 total sources with 10 compact clusters + 10 isolated sources.
    # Use a larger source radius and slightly spread each cluster in space.
    r_source = 5.0
    sources = [
        Source(id=0, pos=np.array([10.0, 10.0, 10.0]), strength=1.0, radius=r_source),
        Source(id=1, pos=np.array([10.0, 12.0, 10.0]), strength=1.0, radius=r_source),
        Source(id=2, pos=np.array([10.0, 10.0, 12.0]), strength=1.0, radius=r_source),
        Source(id=3, pos=np.array([12.0, 10.0, 10.0]), strength=1.0, radius=r_source),
        Source(id=4, pos=np.array([10.0, 10.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=5, pos=np.array([10.0, 12.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=6, pos=np.array([10.0, 10.0, 82.0]), strength=1.0, radius=r_source),
        Source(id=7, pos=np.array([12.0, 10.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=8, pos=np.array([10.0, 80.0, 10.0]), strength=1.0, radius=r_source),
        Source(id=9, pos=np.array([10.0, 82.0, 10.0]), strength=1.0, radius=r_source),
        Source(id=10, pos=np.array([10.0, 80.0, 12.0]), strength=1.0, radius=r_source),
        Source(id=11, pos=np.array([12.0, 80.0, 10.0]), strength=1.0, radius=r_source),
        Source(id=12, pos=np.array([10.0, 80.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=13, pos=np.array([10.0, 82.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=14, pos=np.array([10.0, 80.0, 82.0]), strength=1.0, radius=r_source),
        Source(id=15, pos=np.array([12.0, 80.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=16, pos=np.array([50.0, 20.0, 20.0]), strength=1.0, radius=r_source),
        Source(id=17, pos=np.array([50.0, 22.0, 20.0]), strength=1.0, radius=r_source),
        Source(id=18, pos=np.array([50.0, 20.0, 22.0]), strength=1.0, radius=r_source),
        Source(id=19, pos=np.array([52.0, 20.0, 20.0]), strength=1.0, radius=r_source),
        Source(id=20, pos=np.array([50.0, 20.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=21, pos=np.array([50.0, 22.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=22, pos=np.array([50.0, 20.0, 82.0]), strength=1.0, radius=r_source),
        Source(id=23, pos=np.array([52.0, 20.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=24, pos=np.array([50.0, 80.0, 20.0]), strength=1.0, radius=r_source),
        Source(id=25, pos=np.array([50.0, 82.0, 20.0]), strength=1.0, radius=r_source),
        Source(id=26, pos=np.array([50.0, 80.0, 22.0]), strength=1.0, radius=r_source),
        Source(id=27, pos=np.array([52.0, 80.0, 20.0]), strength=1.0, radius=r_source),
        Source(id=28, pos=np.array([50.0, 80.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=29, pos=np.array([50.0, 82.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=30, pos=np.array([50.0, 80.0, 82.0]), strength=1.0, radius=r_source),
        Source(id=31, pos=np.array([52.0, 80.0, 80.0]), strength=1.0, radius=r_source),
        Source(id=32, pos=np.array([85.0, 30.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=33, pos=np.array([85.0, 32.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=34, pos=np.array([85.0, 30.0, 52.0]), strength=1.0, radius=r_source),
        Source(id=35, pos=np.array([87.0, 30.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=36, pos=np.array([85.0, 70.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=37, pos=np.array([85.0, 72.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=38, pos=np.array([85.0, 70.0, 52.0]), strength=1.0, radius=r_source),
        Source(id=39, pos=np.array([87.0, 70.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=40, pos=np.array([25.0, 50.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=41, pos=np.array([35.0, 35.0, 65.0]), strength=1.0, radius=r_source),
        Source(id=42, pos=np.array([35.0, 65.0, 35.0]), strength=1.0, radius=r_source),
        Source(id=43, pos=np.array([65.0, 35.0, 35.0]), strength=1.0, radius=r_source),
        Source(id=44, pos=np.array([65.0, 65.0, 65.0]), strength=1.0, radius=r_source),
        Source(id=45, pos=np.array([20.0, 50.0, 85.0]), strength=1.0, radius=r_source),
        Source(id=46, pos=np.array([80.0, 15.0, 65.0]), strength=1.0, radius=r_source),
        Source(id=47, pos=np.array([80.0, 50.0, 90.0]), strength=1.0, radius=r_source),
        Source(id=48, pos=np.array([20.0, 85.0, 50.0]), strength=1.0, radius=r_source),
        Source(id=49, pos=np.array([60.0, 50.0, 5.0]), strength=1.0, radius=r_source),
    ]

    if subdomain.rank == 0:
        logger.info("Running domain decompositon with {} ranks.".format(subdomain.comm.Get_size()))

    # Run the domain decomposition
    alps_memory_per_GPU = 96e9 # 96 GB
    ranks_per_GPU = 1
    src_batch_size = 1
    NumTau = 100
    is_periodic_mode_active = True
    subdomain.run_decomposition(global_grid, sources,
                                cost_model = pyC2RayCostModel(max_memory_cost_per_group=alps_memory_per_GPU/ranks_per_GPU,
                                                              source_batch_size=src_batch_size,
                                                              is_periodic_mode_active=is_periodic_mode_active,
                                                              photo_ion_table_size=NumTau),
                                grouping_algorithm="morton",
                                grouping_params = MortonGroupingParams(max_num_sources_per_group=5,
                                                                       morton_bits=10))

    # Collect source groups from all ranks
    source_groups_by_rank: list[list[SourceGroup]] | None = subdomain.comm.gather(
        subdomain.get_source_groups(), root=0
    )

    # Collect local grids from all ranks
    local_grids_by_rank: list[list[RegularGrid]] | None = subdomain.comm.gather(
        subdomain.get_local_grids(), root=0
    )

    # Plot domain decomposition
    if subdomain.rank == 0:
        assert source_groups_by_rank is not None
        assert local_grids_by_rank is not None
        source_groups = [group for rank_groups in source_groups_by_rank for group in rank_groups]
        local_grids = [grid for rank_grids in local_grids_by_rank for grid in rank_grids]

        if not args.skip_npz:
            dump_path = export_domain_decomposition_npz(
                global_grid,
                source_groups,
                local_grids,
                output_path="domain_decomposition_data.npz",
            )
            logger.info("Wrote interactive decomposition dump to %s", dump_path)

        if not args.skip_png:
            plot_domain_decomposition(global_grid, source_groups, local_grids, show=False)

    MPI.Finalize()

if __name__ == "__main__":
    main()
