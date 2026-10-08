"""
This file contains the implementation of the abstract and ASORA raytracing memory
and computational cost models associated to the processing of a group of sources.
"""

from abc import ABC, abstractmethod
from math import ceil, floor
from typing import Any

from pyc2ray import asora_core


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


class ASORACostModel(CostModel):
    """Class providing the implementation of the cost model associated to
    the ASORA raytracing algorithm in GPU mode.

    The memory budget of a group is a fraction of the device memory available when the
    model is built, and it also sets max_memory_cost_per_group: a group is valid as long
    as at least one column density buffer fits in the budget next to its local fields.

    Attributes
    ----------
    max_memory_cost_per_group : float
        Memory budget in bytes for a source group, budget * available_memory.
    max_resident_blocks : int
        Raytracing blocks the device can keep resident, which caps the number of
        sources raytraced concurrently.
    is_periodic_mode_active : bool
        Whether the periodic mode is active in the grid.
    num_column_density_banks : int
        Number of q-shells kept in the ASORA column density ring buffer. If not specified,
        it is retrieved from the ASORA library.
    budget : float
        Fraction of the available device memory to be used for source group processing.
    """

    def __init__(
        self,
        available_memory: int,
        max_resident_blocks: int,
        is_periodic_mode_active: bool,
        num_column_density_banks: int | None = None,
        budget: float = 0.9,
    ) -> None:
        """
        Parameters
        ----------
        available_memory : Available device memory in bytes, queried once by the caller.
        max_resident_blocks : Raytracing blocks the device can keep resident, i.e. the
            maximum number of sources raytraced concurrently, queried once by the caller.
        is_periodic_mode_active : Whether the periodic mode is active in the grid.
        budget : Fraction of the available device memory to be used, in (0, 1].
        """
        super().__init__(budget * available_memory)

        if num_column_density_banks is None:
            num_column_density_banks = asora_core.num_column_density_banks()
        self.num_column_density_banks = num_column_density_banks

        self.max_resident_blocks = max_resident_blocks
        self.is_periodic_mode_active = is_periodic_mode_active
        self.budget = budget

        if not 0.0 < budget <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        if max_resident_blocks < 1:
            raise ValueError(
                f"max_resident_blocks must be >= 1, got {max_resident_blocks}"
            )

    def _compute_cells_to_shell(self, R: float, n_cells_per_side: int) -> int:
        """Compute the total number of cells in the octahedron of q-shells up to q_max
        centered on a source, which is an estimate of the number of cells influenced
        by the source in periodic mode.

        Parameters
        ----------
        R : Maximum radius of source influence in grid units.
        n_cells_per_side : Number of grid cells per side in the local grid.

        Returns
        -------
        The estimated number of cells influenced by the source in periodic mode.
        """

        q_max = asora_core.max_q_shells(R, n_cells_per_side)
        cells_to_shell = (1 + 2 * q_max) * (3 + 2 * q_max * (1 + q_max)) // 3
        return cells_to_shell

    def _compute_cells_in_shell(self, R: float, n_cells_per_side: int) -> int:
        """Compute the number of cells in the outermost q-shell (q_max) around a source,
        which is the size of one bank of the column density ring buffer.

        Parameters
        ----------
        R : Maximum radius of source influence in grid units.
        n_cells_per_side : Number of grid cells per side in the local grid.

        Returns
        -------
        The number of cells in the q_max shell.
        """

        q_max = asora_core.max_q_shells(R, n_cells_per_side)
        if q_max == 0:
            return 1
        return 4 * q_max * q_max + 2

    def compute_group_costs(
        self, R: float, n_cells_per_side: int, n_src: int
    ) -> tuple[float, float]:
        """Compute both memory and computational costs of processing a source group.

        The raytracing of a group needs its local fields and one column density buffer per
        source processed in parallel. The number of sources processed in parallel is
        maximized within the memory budget:

            baseline[bytes] = 8 * 3 * n_cells + 20 * n_src
            coldens[bytes] = 8 * num_banks * cells_in_shell(q_max)
            num_batches = max(
                min(
                    floor((budget - baseline) / coldens),
                    max_resident_blocks,
                    n_src,
                ),
                1,
            )
            M_GPU[bytes] = baseline + num_batches * coldens

            where
                budget = max_memory_cost_per_group
                n_cells = n_cell_per_side^3
                q_max = ceil(min(sqrt(3) * R, n_cell_per_side * 1.5))
                R = maximum radius of source influence in grid units
                num_banks = num_column_density_banks, q-shells kept in the column density ring buffer
                cells_in_shell(q) = 4*q^2 + 2 (1 for q = 0)
                cells_to_shell(q) = (1 + 2*q) * (3 + 2*q*(1 + q)) / 3
                n_src = number of sources in the group

            Terms correspond to:
                - 3 * n_cells doubles: number_density, fraction_HII, photo_ionization_HI
                - n_src doubles + 3*n_src int32: source flux and source positions
                - num_batches * num_banks * cells_in_shell(q_max) doubles: column_density_HI
                  ring buffers (each source processed in parallel keeps only the last
                  num_banks q-shells)

            The photoionization tables are resident on the device before the available
            memory is queried, so they are not part of the group footprint.

        # TODO: define a more accurate computational cost model for the ASORA raytracing algorithm.
        The computational cost is the number of waves of parallel sources needed to process
        the group, ceil(n_src / num_batches), times the number of cells raytraced per source,
        cells_to_shell(q_max), at least in periodic mode.
        # TODO: add the data transfer cost to the computational cost model, which is currently ignored.

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
        if n_src <= 0:
            raise ValueError("At least one source is needed")

        tot_mem = self.max_memory_cost_per_group

        cells_to_shell = self._compute_cells_to_shell(R, n_cells_per_side)
        cells_in_shell = self._compute_cells_in_shell(R, n_cells_per_side)
        n_cells = n_cells_per_side**3
        baseline = (
            # number density
            8 * n_cells
            # average fraction
            + 8 * n_cells
            # photo ionization
            + 8 * n_cells
            # sources (flux and positions)
            + 20 * n_src
        )

        coldens = 8 * self.num_column_density_banks * cells_in_shell
        blocks = floor((tot_mem - baseline) / coldens)

        # At least one batch: when not even one buffer fits in the budget, the memory
        # cost is the minimum footprint, which exceeds the budget, so the grouping
        # rejects the merge. A single-source group is not checked by the grouping, and
        # if it does not fit, ASORA fails when it allocates the buffer.
        n_batches = max(min(blocks, self.max_resident_blocks, n_src), 1)

        mem_cost = baseline + n_batches * coldens

        if self.is_periodic_mode_active:
            comp_cost = cells_to_shell * ceil(n_src / n_batches)
            return mem_cost, comp_cost

        raise NotImplementedError(
            "Computational cost model for non-periodic mode is not implemented."
        )
