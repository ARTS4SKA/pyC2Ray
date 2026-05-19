from dataclasses import dataclass
from typing import List
import numpy as np

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.source_grouping import GroupingParams, SourceGrouping
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import find_enclosing_sphere, evaluate_sphere_intersection

# ================================================================================
# This file contains the MortonGroupingParams and MortonSourceGrouping classes,
# which provide an implementation of the Morton ordering-based grouping algorithm. 
# ===============================================================================


@dataclass
class MortonGroupingParams(GroupingParams):
    """Parameters specific to the Morton grouping algorithm.
    """
    def __init__(self, max_num_sources_per_group: int = 10, morton_bits: int = 10) -> None:
        super().__init__(max_num_sources_per_group)
        self.morton_bits: int = morton_bits

# TODO: split geometric ordeding (Morton-like key) from the actual grouping logic,
# which is more related to the cost model and to the constraints on the groups.
class MortonSourceGrouping(SourceGrouping):
    """Morton ordering-based grouping algorithm.
    """

    def _morton_like_key(self, p: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray, bits: int) -> int:
        """
        Lightweight Morton-like ordering.
        Maps point to integer grid then interleaves bits.

        Parameters
        ----------
        p : np.ndarray
            Point coordinates (shape `(3,)`).
        domain_min : np.ndarray
            Minimum corner of the domain (shape `(3,)`).
        domain_max : np.ndarray
            Maximum corner of the domain (shape `(3,)`).
        bits : int
            Number of bits per dimension for the grid. Total key bits will be 3x this.

        Returns
        -------
        int     Morton-like key for the point.
        """
        # Normalize to [0, 1]
        normalized_position = np.clip((p - domain_min) / np.maximum(domain_max - domain_min, 1e-12), 0.0, 1.0 - 1e-12)

        # Scale to integer by shifting by bits.
        int_position = (normalized_position * (1 << bits)).astype(int)

        def split_by_3(v: int) -> int:
            out = 0
            for i in range(bits):
                out |= ((v >> i) & 1) << (3 * i)
            return out

        return split_by_3(int_position[0]) | (split_by_3(int_position[1]) << 1) | (split_by_3(int_position[2]) << 2)


    def _build_group(self, group_sources: List[Source], grid: Grid, cost_model: CostModel) -> SourceGroup:
        """
        Build a group of sources and compute its geometric and cost properties.

        Parameters
        ----------
        group_sources : List[Source]
            Sources that belong to this group.
        grid : Grid
            Grid used to estimate local cell counts.

        Returns
        -------
        Group     Group object with computed center, radius, bounding box, local cell count, and cost.
        """
        centers = np.array([s.pos for s in group_sources], dtype=float)
        radii = np.array([s.radius for s in group_sources], dtype=float)

        # Find group enclosing sphere and bounding box. The enclosing sphere is used for the radius constraint,
        # while the bounding box is used to estimate the local cell count for cost evaluation.
        if len(group_sources) == 1:
            c = centers[0]
            R = radii[0]
        else:
            c, R = find_enclosing_sphere(centers, radii)
        bbox_min = c - R
        bbox_max = c + R
        # Basic cost evaluation: number of sources times local cell count
        # TODO: this is a very rough estimate. A more accurate cost model could be implemented
        # for example by evaluating the actual raytracing cost for a representative source in the group.
        # TODO: this estimate of n_cells_per_side is not correct in case of non periodic conditions
        n_cells_in_box = grid.find_num_cells_in_box(bbox_min, bbox_max)
        n_cells_per_side = max(1, int(np.ceil(n_cells_in_box ** (1.0 / 3.0))))
        mem_cost, comp_cost = cost_model.compute_group_costs(
            group_sources[0].radius,
            n_cells_per_side,
            len(group_sources),
        )

        return SourceGroup(
            id=-1,  # ID will be assigned later
            sources=list(group_sources),
            center=c,
            radius=R,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            mem_cost=mem_cost,
            comp_cost=comp_cost
        )

    def build_groups(self, sources: List[Source], grid: Grid, grouping_params: GroupingParams, cost_model: CostModel) -> List[SourceGroup]:
        """Build the groups of sources to be assigned to the ranks using Morton ordering.

        Parameters
        ----------
        sources : List[Source]
            List of sources in the provided grid.
        grid : Grid
            The grid of the simulation. (can be a sub-grid, in case of recursive grouping)
        grouping_params : GroupingParams
            The parameters for the Morton grouping algorithm. Must be an
            instance of MortonGroupingParams.
        cost_model : CostModel
            The cost model to use for the evaluation of the cost of processing a group of sources.

        Returns
        -------
        List[SourceGroup]
            The list of source groups to be assigned to the ranks.
        """
        if not isinstance(grouping_params, MortonGroupingParams):
            raise TypeError("Morton grouping requires MortonGroupingParams.")

        if not sources:
            return []

        # Compute spatial ordering
        ordered_sources = sorted(sources, key = lambda s: self._morton_like_key(s.pos, grid.get_domain_min(), grid.get_domain_max(), grouping_params.morton_bits))

        def valid(g: SourceGroup) -> bool:
            return (
                len(g.sources) <= grouping_params.max_num_sources_per_group
                and g.mem_cost <= cost_model.max_memory_cost_per_group
            )

        # TODO: optimize the loop below.
        # For example: avoid rebuilding the same single source group at loop end when gtrial already matches current_group
        # (last-source rejection/non-intersection path).
        source_groups: List[SourceGroup] = []
        current_group: List[Source] = []
        for s in ordered_sources:
            if not current_group:
                current_group = [s]
                gtrial = self._build_group(current_group, grid, cost_model)
                continue

            # Check if the new source intersects with the current group. If not, we can start a new group.
            if not evaluate_sphere_intersection(gtrial.center, gtrial.radius, s.pos, s.radius):
                source_groups.append(gtrial)
                current_group = [s]
                gtrial = self._build_group(current_group, grid, cost_model)
                continue

            # If the new source intersects with the current group
            # we try to add it to the group and check if it's still valid.
            trial = current_group + [s]
            gtrial = self._build_group(trial, grid, cost_model)

            if valid(gtrial):
                current_group = trial
            else:
                source_groups.append(self._build_group(current_group, grid, cost_model))
                current_group = [s]
                gtrial = self._build_group(current_group, grid, cost_model)

        if current_group:
            source_groups.append(self._build_group(current_group, grid, cost_model))

        # Update group IDs
        for i, g in enumerate(source_groups):
            g.id = i

        return source_groups

