"""
This file contains the MortonGroupingParams and MortonSourceGrouping classes,
which provide an implementation of the Morton ordering-based grouping algorithm.
"""

from dataclasses import dataclass

import numpy as np

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.grid import Grid
from pyc2ray.domain.source_grouping import GroupingParams, SourceGrouping
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.utils import (
    evaluate_sphere_intersection,
    expand_enclosing_sphere,
    find_enclosing_sphere,
)


@dataclass
class MortonGroupingParams(GroupingParams):
    """Parameters specific to the Morton grouping algorithm."""

    morton_bits: int = 10


# TODO: split geometric ordeding (Morton-like key) from the actual grouping logic,
# which is more related to the cost model and to the constraints on the groups.
class MortonSourceGrouping(SourceGrouping):
    """Morton ordering-based grouping algorithm."""

    def _morton_like_key(
        self, p: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray, bits: int
    ) -> int:
        """
        Lightweight Morton-like ordering.
        Maps point to integer grid then interleaves bits.

        Parameters
        ----------
        p : Point coordinates (shape `(3,)`).
        domain_min : Minimum corner of the domain (shape `(3,)`).
        domain_max : Maximum corner of the domain (shape `(3,)`).
        bits : Number of bits per dimension for the grid. Total key bits will be 3x this.

        Returns
        -------
        Morton-like key for the point.
        """
        # Normalize input coordinates to [0, 1]
        normalized_position = np.clip(
            (p - domain_min) / np.maximum(domain_max - domain_min, 1e-12),
            0.0,
            1.0 - 1e-12,
        )

        # Scale to integer by shifting by bits: the normalized_position input position is a 3 element array
        # of floating point numbers in [0, 1], so multiplying them by 2^bits gives an integer in [0, 2^bits)
        # when truncated. THe larger the bits, the finer the spatial resolution of the Morton ordering, but
        # also the larger the resulting keys (which can affect performance and memory usage).
        int_position = (normalized_position * (1 << bits)).astype(int)

        # Interleave bits to get the Morton key. The interleaving is done by taking the bits of each coordinate
        # and placing them in the final key in an interleaved manner.
        # TODO: the loop over bits for each coordinate for each source can become a hotspot when ordering large
        # source lists. Consider using a faster bit-interleaving approach (e.g., precomputed lookup tables per byte/word,
        # vectorized numpy where practical, or a specialized Morton encoding routine) to reduce per-source overhead during sorting.
        def split_by_3(v: int) -> int:
            out = 0
            for i in range(bits):
                out |= ((v >> i) & 1) << (3 * i)
            return out

        # The final Morton key is obtained by interleaving the bits of the x, y, and z coordinates.
        # For example, if bits=10, we take the 10 bits of the x coordinate and place them in positions 0, 3, 6, ...,
        # the 10 bits of the y coordinate and place them in positions 1, 4, 7, ..., and the 10 bits of the z coordinate
        # and place them in positions 2, 5, 8, ... of the final key.
        return (
            split_by_3(int_position[0])
            | (split_by_3(int_position[1]) << 1)
            | (split_by_3(int_position[2]) << 2)
        )

    def _group_costs(
        self,
        center: np.ndarray,
        radius: float,
        first_source_radius: float,
        n_sources: int,
        grid: Grid,
        cost_model: CostModel,
    ) -> tuple[float, float]:
        """Evaluate the memory and computational cost of a group from its enclosing sphere.

        Parameters
        ----------
        center : Center of the enclosing sphere of the group.
        radius : Radius of the enclosing sphere of the group.
        first_source_radius : Radius of influence of the first source of the group.
        n_sources : Number of sources in the group.
        grid : Grid used to estimate local cell counts.
        cost_model : The cost model used to evaluate the cost of processing the group.

        Returns
        -------
        Memory and computational cost of the group.
        """
        # Basic cost evaluation: number of sources times local cell count
        # TODO: this is a very rough estimate. A more accurate cost model could be implemented
        # for example by evaluating the actual raytracing cost for a representative source in the group,
        # by taking care of the different source radii, etc...
        # TODO: this estimate of n_cells_per_side is not correct in case of non periodic conditions
        n_cells_in_box = grid.find_num_cells_in_box(center - radius, center + radius)
        n_cells_per_side = max(1, int(np.ceil(n_cells_in_box ** (1.0 / 3.0))))
        # The cost model expects the radius of influence in grid units, while the source radius is
        # stored as a physical length. Convert it using the local cell size around the group center
        # (constant for regular grids, position dependent for non-uniform grids such as AMR).
        # TODO: with heterogeneous source radii the first source is not representative of the
        # group.
        radius_in_grid_units = first_source_radius / grid.get_average_cell_size(center)
        return cost_model.compute_group_costs(
            radius_in_grid_units,
            n_cells_per_side,
            n_sources,
        )

    def _build_group(
        self, group_sources: list[Source], grid: Grid, cost_model: CostModel
    ) -> SourceGroup:
        """
        Build a group of sources and compute its geometric and cost properties.

        Parameters
        ----------
        group_sources : Sources that belong to this group.
        grid : Grid used to estimate local cell counts.

        Returns
        -------
        Source group with computed center, radius, bounding box, local cell count, and cost.
        """
        centers = np.array([s.pos for s in group_sources], dtype=float)
        radii = np.array([s.radius for s in group_sources], dtype=float)

        # Find group enclosing sphere and bounding box. The enclosing sphere is used for the radius constraint,
        # while the bounding box is used to estimate the local cell count for cost evaluation.
        c, R = find_enclosing_sphere(centers, radii)
        bbox_min = c - R
        bbox_max = c + R
        mem_cost, comp_cost = self._group_costs(
            c, R, group_sources[0].radius, len(group_sources), grid, cost_model
        )

        return SourceGroup(
            id=-1,  # ID will be assigned later
            sources=list(group_sources),
            center=c,
            radius=R,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            mem_cost=mem_cost,
            comp_cost=comp_cost,
        )

    def build_groups_incremental(
        self,
        sources: list[Source],
        grid: Grid,
        grouping_params: GroupingParams,
        cost_model: CostModel,
    ) -> list[SourceGroup]:
        """Build source groups keeping the enclosing sphere up to date incrementally.

        The enclosing incremental sphere is only an upper bound on the best fitting one,
        so it is used for the split and validity decisions only: each group is re-fitted
        accurately once it is closed, which costs one fit per group instead of one per source.

        Because the decisions are taken on an upper bound rather than on an accurate
        sphere, the split points differ from the ones an accurate fit would produce,
        through two opposing effects. The intersection test is more permissive, since a
        larger sphere intersects the next source more often, so geometric splits can be
        missed and sources merged into a group an accurate sphere would have left out.
        The validity test is instead more conservative, since the cost is evaluated on an
        inflated sphere, so a group can be closed while an accurate fit would still have
        accepted the candidate. Which one prevails depends on the source distribution.
        The constraints themselves are always enforced, and the geometry stored on a
        closed group is never looser than an accurate fit of the same members, since the
        tighter of the two spheres is kept.

        Parameters
        ----------
        sources : List of sources in the provided grid.
        grid : The grid of the simulation. (can be a sub-grid, in case of recursive grouping)
        grouping_params : The parameters for the Morton grouping algorithm. Must be an
        instance of MortonGroupingParams.
        cost_model : The cost model to use for the evaluation of the cost of processing a group of sources.

        Returns
        -------
        The list of source groups to be assigned to the ranks.
        """
        if not isinstance(grouping_params, MortonGroupingParams):
            raise TypeError("Morton grouping requires MortonGroupingParams.")

        if not sources:
            return []

        # Compute spatial ordering
        ordered_sources = sorted(
            sources,
            key=lambda s: self._morton_like_key(
                s.pos,
                grid.get_domain_min(),
                grid.get_domain_max(),
                grouping_params.morton_bits,
            ),
        )

        max_num_sources = grouping_params.max_num_sources_per_group
        max_mem_cost = cost_model.max_memory_cost_per_group

        # Scratch buffer reused for every call that needs a center as an array: allocating
        # a fresh one per candidate source would show up in the profile.
        scratch_center = np.empty(3, dtype=float)

        def close_group(
            members: list[Source], cx: float, cy: float, cz: float, radius: float
        ) -> SourceGroup:
            """Finalize a group, re-fitting its enclosing sphere accurately.

            The incremental sphere is an upper bound and depends on the merge order, while
            the fit is approximate, so the tighter of the two is kept: a loose radius here
            would inflate the local grid built around the group for the whole simulation.
            """
            center = np.array((cx, cy, cz), dtype=float)
            if len(members) > 1:
                fitted_center, fitted_radius = find_enclosing_sphere(
                    np.array([m.pos for m in members], dtype=float),
                    np.array([m.radius for m in members], dtype=float),
                )
                if fitted_radius < radius:
                    center, radius = fitted_center, fitted_radius

            mem_cost, comp_cost = self._group_costs(
                center, radius, members[0].radius, len(members), grid, cost_model
            )
            return SourceGroup(
                id=-1,  # ID will be assigned later
                sources=list(members),
                center=center,
                radius=radius,
                bbox_min=center - radius,
                bbox_max=center + radius,
                mem_cost=mem_cost,
                comp_cost=comp_cost,
            )

        source_groups: list[SourceGroup] = []
        members = [ordered_sources[0]]
        center_x, center_y, center_z = (
            float(members[0].pos[0]),
            float(members[0].pos[1]),
            float(members[0].pos[2]),
        )
        group_radius = float(members[0].radius)

        for s in ordered_sources[1:]:
            # Check if the new source intersects with the current group. If not, we can start a new group.
            scratch_center[0] = center_x
            scratch_center[1] = center_y
            scratch_center[2] = center_z
            intersects = evaluate_sphere_intersection(
                scratch_center, group_radius, s.pos, s.radius
            )

            if intersects:
                # If the new source intersects with the current group we try to add it to
                # the group and check if the resulting group would still be valid. Both the
                # trial sphere and its cost are obtained without fitting anything.
                trial_x, trial_y, trial_z, trial_radius = expand_enclosing_sphere(
                    center_x,
                    center_y,
                    center_z,
                    group_radius,
                    float(s.pos[0]),
                    float(s.pos[1]),
                    float(s.pos[2]),
                    float(s.radius),
                )
                n_trial_sources = len(members) + 1
                accepted = n_trial_sources <= max_num_sources
                if accepted:
                    scratch_center[0] = trial_x
                    scratch_center[1] = trial_y
                    scratch_center[2] = trial_z
                    trial_mem_cost, _ = self._group_costs(
                        scratch_center,
                        trial_radius,
                        members[0].radius,
                        n_trial_sources,
                        grid,
                        cost_model,
                    )
                    accepted = trial_mem_cost <= max_mem_cost

                if accepted:
                    members.append(s)
                    center_x, center_y, center_z = trial_x, trial_y, trial_z
                    group_radius = trial_radius
                    continue

            source_groups.append(
                close_group(members, center_x, center_y, center_z, group_radius)
            )
            members = [s]
            center_x, center_y, center_z = (
                float(s.pos[0]),
                float(s.pos[1]),
                float(s.pos[2]),
            )
            group_radius = float(s.radius)

        source_groups.append(
            close_group(members, center_x, center_y, center_z, group_radius)
        )

        # Update group IDs
        for i, g in enumerate(source_groups):
            g.id = i

        return source_groups

    # TODO: check source ordering influence on group creation: ideally, the order of sources in the input list should not influence the final groups,
    # which should only depend on their spatial distribution and on the cost model.
    def build_groups(
        self,
        sources: list[Source],
        grid: Grid,
        grouping_params: GroupingParams,
        cost_model: CostModel,
    ) -> list[SourceGroup]:
        """Build the groups of sources to be assigned to the ranks using Morton ordering.

        Parameters
        ----------
        sources : List of sources in the provided grid.
        grid : The grid of the simulation. (can be a sub-grid, in case of recursive grouping)
        grouping_params : The parameters for the Morton grouping algorithm. Must be an
        instance of MortonGroupingParams.
        cost_model : The cost model to use for the evaluation of the cost of processing a group of sources.

        Returns
        -------
        The list of source groups to be assigned to the ranks.
        """
        if not isinstance(grouping_params, MortonGroupingParams):
            raise TypeError("Morton grouping requires MortonGroupingParams.")

        if not sources:
            return []

        # Compute spatial ordering
        ordered_sources = sorted(
            sources,
            key=lambda s: self._morton_like_key(
                s.pos,
                grid.get_domain_min(),
                grid.get_domain_max(),
                grouping_params.morton_bits,
            ),
        )

        def valid(g: SourceGroup) -> bool:
            return (
                len(g.sources) <= grouping_params.max_num_sources_per_group
                and g.mem_cost <= cost_model.max_memory_cost_per_group
            )

        source_groups: list[SourceGroup] = []
        current_group: list[Source] = [ordered_sources[0]]
        gtrial = self._build_group(current_group, grid, cost_model)
        for s in ordered_sources[1:]:
            # Check if the new source intersects with the current group. If not, we can start a new group.
            if not evaluate_sphere_intersection(
                gtrial.center, gtrial.radius, s.pos, s.radius
            ):
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
