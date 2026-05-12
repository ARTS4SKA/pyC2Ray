from __future__ import annotations

import mpi4py
mpi4py.rc.finalize = False
from mpi4py import MPI
import numpy as np
import re
from pathlib import Path

from pyc2ray.domain.morton_grouping import MortonGroupingParams
from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain
from pyc2ray.domain.utils import get_domain_logger
from pyc2ray.visualization.domain_decomposition import (
    export_domain_decomposition_npz,
    plot_domain_decomposition,
)

logger = get_domain_logger(__name__)

SOURCE_FILENAME_RE = re.compile(r"CDM_100Mpc_2048\.(\d{5})\.halo\.txt$")

def _snapshot_id_from_name(name: str) -> int:
    match = SOURCE_FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid source filename: {name}")
    return int(match.group(1))

def get_source_file_paths(source_dir: str | Path) -> list[Path]:
    """Return sorted source files matching CDM_100Mpc_2048.XXXXX.halo.txt."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    file_paths: list[Path] = []
    for p in source_dir.glob("CDM_100Mpc_2048.*.halo.txt"):
        if SOURCE_FILENAME_RE.match(p.name):
            file_paths.append(p)

    file_paths.sort(key=lambda p: _snapshot_id_from_name(p.name))
    return file_paths

def load_source_positions(source_file: str | Path) -> np.ndarray:
    """Load source positions from columns 2, 3 and 4 (1-based counting).

    Returns
    -------
    np.ndarray
        Array of shape (num_sources, 3) containing x, y, z positions as floats.
    """
    source_file = Path(source_file)
    # usecols are 0-based -> (1, 2, 3) corresponds to 2nd, 3rd, 4th columns
    positions = np.loadtxt(source_file, usecols=(1, 2, 3), dtype=np.float64)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
    return positions

def load_source_positions_set(source_dir: str | Path) -> np.ndarray:
    """Load and merge all matching source files for one split snapshot.

    Returns
    -------
    np.ndarray
        Combined source positions of shape (num_sources_total, 3).
    """
    file_paths = get_source_file_paths(source_dir)
    if not file_paths:
        raise FileNotFoundError(
            f"No source files matching CDM_100Mpc_2048.XXXXX.halo.txt in {source_dir}"
        )


    chunks = [load_source_positions(p) for p in file_paths[:1]]  # limit to first 2 files for testing
    return np.vstack(chunks)


def main():
    """Run the domain decomposition step and plot the resulting source groups and local grids."""

    # Create subdomain handler
    subdomain = Subdomain(MPI.COMM_WORLD)

    # Load source positions from files
    source_positions = load_source_positions_set(
        "/capstor/store/cscs/pasc/c45/CDM_100Mpc_2048/sources/"
    )

    # Prepare Source objects with dummy strength and radius
    sources: list[Source] = []
    for i, pos in enumerate(source_positions):
        sources.append(Source(id=i, pos=pos, strength=1.0, radius=1.0))

    if subdomain.rank == 0:
        logger.info("Running domain decompositon with {} ranks.".format(subdomain.comm.Get_size()))

    # Create the global grid: for this example we are defining the global size
    # based on the loaded source position and the cell size is arbitrary (e.g. 1 Mpc)
    cell_size = 1.0
    max_coord = float(np.max(source_positions))
    num_cells = int(np.ceil((max_coord + 0.5 * cell_size) / cell_size))
    global_grid = RegularGrid(cell_size=cell_size, num_cells=num_cells)
    if subdomain.rank == 0:
        logger.info("Given the cell size {}, the global grid has {} cells.".format(cell_size, num_cells))

    # Run the domain decomposition
    subdomain.run_decomposition(global_grid, sources, grouping_algorithm="morton", 
                                grouping_params = MortonGroupingParams(max_num_sources_per_group=5,
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

    # Plot domain decomposition
    if subdomain.rank == 0:
        assert source_groups_by_rank is not None
        assert local_grids_by_rank is not None
        source_groups = [group for rank_groups in source_groups_by_rank for group in rank_groups]
        local_grids = [grid for rank_grids in local_grids_by_rank for grid in rank_grids]

        dump_path = export_domain_decomposition_npz(
            global_grid,
            source_groups,
            local_grids,
            output_path="domain_decomposition_data.npz",
        )
        logger.info("Wrote interactive decomposition dump to %s", dump_path)

        plot_domain_decomposition(global_grid, source_groups, local_grids, show=False)

    MPI.Finalize()

if __name__ == "__main__":
    main()
