from dataclasses import dataclass
from typing import List
import numpy as np

from pyc2ray.domain.source_grouping import GroupingParams, SourceGrouping
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import find_enclosing_sphere

# ================================================================================
# This file contains the MortonGroupingParams and MortonSourceGrouping classes,
# which provide an implementation of the Morton ordering-based grouping algorithm. 
# ===============================================================================


@dataclass
class MortonGroupingParams(GroupingParams):
    """Parameters specific to the Morton grouping algorithm.
    """
    def __init__(self, max_num_sources_per_group: int = 10, max_cost_per_group: float = 1.0,
                 morton_bits: int = 10) -> None:
        super().__init__(max_num_sources_per_group, max_cost_per_group)
        self.morton_bits: int = morton_bits

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

    def _evaluate_group(self, group_sources: List[Source], grid: Grid) -> SourceGroup:
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
        c, R = find_enclosing_sphere(centers, radii)
        bbox_min = c - R
        bbox_max = c + R
        # Basic cost evaluation: number of sources times local cell count
        # TODO: this is a very rough estimate. A more accurate cost model could be implemented.
        cost = len(group_sources) * grid.find_num_cells_in_box(bbox_min, bbox_max)

        return SourceGroup(
            id=-1,  # ID will be assigned later
            sources=list(group_sources),
            center=c,
            radius=R,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            cost=cost,
        )

    def build_groups(self, sources: List[Source], grid: Grid, grouping_params: GroupingParams) -> List[SourceGroup]:
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
                and g.cost <= grouping_params.max_cost_per_group
            )

        source_groups: List[SourceGroup] = []
        current_group: List[Source] = []
        for s in ordered_sources:
            if not current_group:
                current_group = [s]
                continue

            trial = current_group + [s]
            gtrial = self._evaluate_group(trial, grid)

            if valid(gtrial):
                current_group = trial
            else:
                source_groups.append(self._evaluate_group(current_group, grid))
                current_group = [s]

        if current_group:
            source_groups.append(self._evaluate_group(current_group, grid))

        return source_groups

