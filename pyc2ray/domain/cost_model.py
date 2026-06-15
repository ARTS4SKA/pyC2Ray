"""
This file contains the implementation of the abstract and pyC2Ray memory
and computational cost models associated to the processing of a group of
sources.
"""

from abc import ABC, abstractmethod
from math import sqrt
from typing import Any

import numpy as np


class CostModel(ABC):
    """Abstract class providing a common interface for memory/computational cost models associated
    to source group processing.

    Attributes
    ----------
    max_memory_cost_per_group : Maximum (model dependent) allowed memory cost for a source group,
    used as a threshold in grouping algorithms.
    """

    def __init__(self, max_memory_cost_per_group: float) -> None:
        self.max_memory_cost_per_group = max_memory_cost_per_group

    @abstractmethod
    def compute_group_costs(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Compute both memory and computational costs of processing a source group.

        Parameters
        ----------
        Notes
        -----
        Different concrete cost models may require different inputs, so this
        abstract interface intentionally stays generic. Concrete subclasses
        should document and type-annotate their own expected arguments.

        Returns
        -------
        The computed memory and computational costs of processing the group.
        """
        pass


class pyC2RayCostModel(CostModel):
    """Class providing the implementation of the cost model associated to
    the pyC2Ray raytracing algorithm in GPU mode.

    Attributes
    ----------
    max_memory_cost_per_group : float
        Maximum allowed memory cost for a source group, used as a threshold in grouping algorithms.
    source_batch_size : int
        Number of sources to be processed in parallel.
    is_periodic_mode_active : bool
        Whether the periodic mode is active in the grid.
    photo_ion_table_size : int
        Size of the phi_ion table.
    """

    def __init__(
        self,
        max_memory_cost_per_group: float,
        source_batch_size: int,
        is_periodic_mode_active: bool,
        photo_ion_table_size: int,
    ) -> None:
        super().__init__(max_memory_cost_per_group)
        self.source_batch_size = source_batch_size
        self.photo_ion_table_size = photo_ion_table_size
        self.is_periodic_mode_active = is_periodic_mode_active

    def _compute_cells_to_shell(self, R: float, n_cells_per_side: int) -> int:
        """Compute the total number of cells in the box of side 2*R centered on a source,
        which is an estimate of the number of cells influenced by the source in periodic mode.

        Parameters
        ----------
        R : Maximum radius of source influence in grid units.
        n_cells_per_side : Number of grid cells per side in the local grid.

        Returns
        -------
        The estimated number of cells influenced by the source in periodic mode.
        """

        if n_cells_per_side <= 0:
            raise ValueError("n_cells_per_side must be > 0")
        if R < 0:
            raise ValueError("R must be >= 0")

        # TODO: these calculations are already present in multiple places,
        # refactoring is needed
        q_max = int(np.ceil(sqrt(3) * min(R, sqrt(3) * n_cells_per_side / 2)))
        cells_to_shell = (1 + 2 * q_max) * (3 + 2 * q_max * (1 + q_max)) // 3
        return cells_to_shell

    def compute_group_costs(
        self, R: float, n_cells_per_side: int, n_src: int
    ) -> tuple[float, float]:
        """Compute both memory and computational costs of processing a source group.

        The GPU memory footprint for a single source group of n_src sources, to be processed in batches of size batch_size,
        is given by:

            M_GPU[bytes] =
                8 * (3 * n_cells + batch_size * cells_to_shell(q_max) + 2 * photo_ion_table_size + n_src) + 12 * n_src

            where
                n_cells = n_cell_per_side^3
                q_max = ceil(sqrt(3) * min(R, sqrt(3) * n_cell_per_side / 2))
                R = maximum radius of source influence in grid units
                cells_to_shell(q) = (1 + 2*q) * (3 + 2*q*(1 + q)) / 3
                photo_ion_table_size = number of entries in each photoionization table
                n_src = number of sources in the group

            Terms correspond to:
                - 3 * n_cells doubles: number_density, fraction_HII, photo_ionization_HI
                - batch_size * cells_to_shell(q_max) doubles: column_density_HI
                - 2 * photo_ion_table_size doubles: thin/thick photoionization tables
                - n_src doubles + 3*n_src int32: source flux and source positions


        # TODO: define a more accurate computational cost model for the pyC2Ray raytracing algorithm.
        Concerning the computational cost, for the pyC2Ray raytracing algorithm, we can temporarily use a simple cost model
        where the cost is proportional to the number of sources in the group and the number of cells influenced by the
        sources at least in periodic mode.

        It is assumed that all the sources in the group have the same radius of influence R and that the
        local grid is regular and cubic.


        Parameters
        ----------
        R : Maximum radius of source influence in grid units.
        n_cells_per_side : Number of grid cells per side in the local grid.
        n_src : Number of sources in the group.

        Returns
        -------
        The estimated memory and computational costs of processing the group.
        """
        if n_src < 0:
            raise ValueError("n_src must be >= 0")

        cells_to_shell = self._compute_cells_to_shell(R, n_cells_per_side)
        n_cells = n_cells_per_side**3
        mem_cost = (
            8
            * (
                3 * n_cells
                + self.source_batch_size * cells_to_shell
                + 2 * self.photo_ion_table_size
                + n_src
            )
            + 12 * n_src
        )

        if self.is_periodic_mode_active:
            comp_cost = n_src * min(n_cells, cells_to_shell)
            return mem_cost, comp_cost

        raise NotImplementedError(
            "Computational cost model for non-periodic mode is not implemented."
        )
