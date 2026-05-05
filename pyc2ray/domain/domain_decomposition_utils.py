from __future__ import annotations

from typing import List, Tuple
import numpy as np

from pyc2ray.domain.utils import get_domain_logger

logger = get_domain_logger(__name__)

# TODO CB: consolidate or delete
def log_domain_decomposition_assignments(
    ranks_groups: List[List[Group]],
    ranks_costs: List[float],
    dr: float = 0.0,
) -> None:
    """Log domain decomposition source group data and distribution for all ranks.

    This function is intended to be called by rank 0 after assignment.
    """
    for rank, groups in enumerate(ranks_groups):
        n_local_groups = len(groups)
        n_local_sources = sum(len(g.sources) for g in groups)
        rank_cost = float(ranks_costs[rank])

        logger.info(
            "Scatter check | rank=%d groups=%d sources=%d",
            rank,
            n_local_groups,
            n_local_sources,
        )

        for i, g in enumerate(groups):
            if dr > 0.0:
                logger.info(
                    (
                        "Local group index=%d cost=%.3e "
                        "center=(%.2f, %.2f, %.2f) "
                        "center in cell units=(%.2f, %.2f, %.2f) "
                        "radius=%.2f radius in cell units=(%.2f) "
                        "bounding_box_min=(%d, %d, %d) "
                        "bounding_box_max=(%d, %d, %d) "
                        "clipped_bounding_box_min=(%d, %d, %d) "
                        "clipped_bounding_box_max=(%d, %d, %d)"
                    ),
                    i,
                    rank_cost,
                    g.center[0],
                    g.center[1],
                    g.center[2],
                    g.center[0] / dr,
                    g.center[1] / dr,
                    g.center[2] / dr,
                    g.radius,
                    g.radius / dr,
                    g.cells[0][0] + 1,
                    g.cells[0][1] + 1,
                    g.cells[0][2] + 1,
                    g.cells[1][0] + 1,
                    g.cells[1][1] + 1,
                    g.cells[1][2] + 1,
                    g.cells[2][0] + 1,
                    g.cells[2][1] + 1,
                    g.cells[2][2] + 1,
                    g.cells[3][0] + 1,
                    g.cells[3][1] + 1,
                    g.cells[3][2] + 1,
                )
            else:
                logger.info(
                    (
                        "Local group index=%d cost=%.3e "
                        "center=(%.2f, %.2f, %.2f) "
                        "bounding_box_min=(%d, %d, %d) "
                        "bounding_box_max=(%d, %d, %d) "
                        "clipped_bounding_box_min=(%d, %d, %d) "
                        "clipped_bounding_box_max=(%d, %d, %d)"
                    ),
                    i,
                    rank_cost,
                    g.center[0],
                    g.center[1],
                    g.center[2],
                    g.cells[0][0] + 1,
                    g.cells[0][1] + 1,
                    g.cells[0][2] + 1,
                    g.cells[1][0] + 1,
                    g.cells[1][1] + 1,
                    g.cells[1][2] + 1,
                    g.cells[2][0] + 1,
                    g.cells[2][1] + 1,
                    g.cells[2][2] + 1,
                    g.cells[3][0] + 1,
                    g.cells[3][1] + 1,
                    g.cells[3][2] + 1,
                )


    logger.info("Source groups assigned to ranks.")


def overlap_volume(a_min, a_max, b_min, b_max) -> float:
    """Return intersection volume between two axis-aligned boxes.

    Parameters
    ----------
    a_min : np.ndarray
        Minimum corner of box A (shape `(3,)`).
    a_max : np.ndarray
        Maximum corner of box A (shape `(3,)`).
    b_min : np.ndarray
        Minimum corner of box B (shape `(3,)`).
    b_max : np.ndarray
        Maximum corner of box B (shape `(3,)`).

    Returns
    -------
    float
        Intersection volume between the two boxes.
    """
    if np.any(a_max <= a_min) or np.any(b_max <= b_min):
        raise ValueError('Invalid box: max corners must be "greater" than min corner.')
    overlap_min = np.maximum(a_min, b_min)
    overlap_max = np.minimum(a_max, b_max)
    d = np.maximum(0.0, overlap_max - overlap_min)
    return float(d[0] * d[1] * d[2])


class Grid:
    """ Utility class for grid-related calculations, such as finding overlapping cells for a given box.

    The grid is assumed to have its "origin" at (0, 0, 0), a cubic shape and be axis-aligned. Moreover
    the grid is represented in physical coordinates, i.e. the cell size is given by `dx`.

    Attributes
    ----------
    num_cells : int
        Number of cells in each dimension.
    dx : float
        Uniform cell size for the grid.
    """
    num_cells: int
    dx: float

    def __init__(
        self,
        num_cells: int,
        dx: float,
    ) -> None:

        self.num_cells = num_cells
        self.dx = dx

        if self.num_cells <= 0:
            raise ValueError("Invalid number of cells: must be a positive integer.")
        if self.dx <= 0.0:
            raise ValueError("Invalid cell size: dx must be positive.")

    def find_cells_in_box(self, box_min: np.ndarray, box_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Estimate cell count and corresponding grid indexes in a box.
        The box is not limited to the grid domain, but only the overlapping part is considered for cell counting
        since the "external" is not included in pyC2Ray computations.

        Parameters
        ----------
        box_min : np.ndarray
            Minimum corner of the axis-aligned box (shape `(3,)`).
        box_max : np.ndarray
            Maximum corner of the axis-aligned box (shape `(3,)`).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
            Overlapping cell group.
            The tuple contains (cell_min_indexes, cell_max_indexes, cell_min_indexes_clipped, cell_max_indexes_clipped, effective_volume),
            where cell indexes are 0-based (beware of discrepancies wrt C2Ray). cell_max_indexes are inclusive.
        """
        if np.any(box_max <= box_min):
            raise ValueError("Invalid box: box_max must be greater than box_min.")

        vol = overlap_volume(box_min, box_max, np.zeros(3), np.array(self.num_cells) * self.dx)
        if vol > 0.0:
            # This calculation already accounts for the fact that the box may be partially outside the grid domain
            min_indexes = np.floor(box_min / self.dx).astype(int)
            max_indexes = np.ceil(box_max / self.dx).astype(int) - 1
            # Compute the effective volume contained in the grid domain
            min_indexes_clipped = np.maximum(min_indexes, 0)
            max_indexes_clipped = np.minimum(max_indexes, self.num_cells - 1)
            effective_volume = np.prod((max_indexes_clipped - min_indexes_clipped + 1))*(self.dx ** 3)
            return (min_indexes, max_indexes, min_indexes_clipped, max_indexes_clipped, effective_volume)

        return (np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros(3, dtype=int), 0)

    def get_domain_min(self) -> np.ndarray:
        """Return the minimum corner of the grid domain."""
        return np.zeros(3)

    def get_domain_max(self) -> np.ndarray:
        """Return the maximum corner of the grid domain."""
        return np.array(self.num_cells) * self.dx

class Source:
    """Represents a single radiation source.

    Attributes
    ----------
    gid : int
        Global source identifier.
    pos : np.ndarray
        Source position in 3D Cartesian coordinates (shape `(3,)`).
    strength : float
        Source emissivity/intensity used by downstream physics models.
    radius : float
        Maximum tracing/influence radius for this source.
    """
    def __init__(
        self,
        gid: int,
        pos: np.ndarray,
        strength: float,
        radius: float,
    ) -> None:
        self.gid = gid
        self.pos = pos
        self.strength = strength
        self.radius = radius


class Group:
    """Container for a set of sources grouped for joint processing.

    Attributes
    ----------
    sources : List[Source]
        Sources that belong to this group.
    center : np.ndarray
        Center of the enclosing sphere of grouped source spheres (shape `(3,)`).
    radius : float
        Radius of the enclosing sphere.
    bbox_min : np.ndarray
        Minimum corner of the cubic axis-aligned bounding box around the group.
    bbox_max : np.ndarray
        Maximum corner of the cubic axis-aligned bounding box around the group.
    cells : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        Overlapping cell group. The tuple contains (cell_min_indexes, cell_max_indexes, cell_min_indexes_clipped, cell_max_indexes_clipped, effective_volume),
        where cell indexes are 0-based (beware of discrepancies wrt C2Ray). cell_max_indexes are inclusive.
    cost : float
        Estimated computational cost for this group.
    """
    def __init__(
        self,
        sources: List[Source],
        center: np.ndarray,
        radius: float,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        cells: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
        cost: float,
    ) -> None:
        self.sources = list(sources)
        self.center = center.copy()
        self.radius = radius
        self.bbox_min = bbox_min.copy()
        self.bbox_max = bbox_max.copy()
        self.cost = cost
        self.cells = cells

    def get_source_ids(self) -> List[int]:
        """Return the list of source global IDs in this group."""
        return [s.gid for s in self.sources]

    def get_num_cells(self) -> int:
        """Return the total number of domain (only included) cells covered by this group bounding box."""
        if self.cells[4] == 0:
            # Grid volume for this group is zero.
            return 0
        else:
            # cell_max_indexes are inclusive, so we need to add 1 to get the count of cells along each axis.
            return np.prod((self.cells[3] - self.cells[2]) + 1)

    def get_num_cells_per_side(self) -> int:
        """Return the number of domain (only included) cells per side for the cubic bounding box."""
        if self.cells[4] == 0:
            # Grid volume for this group is zero, so we can consider it as having zero cells per side.
            return 0
        else:
            # cell_max_indexes are inclusive, so we need to add 1 to get the count of cells along each axis.
            return int(self.cells[3][0] - self.cells[2][0] + 1)

    def get_full_num_cells_per_side(self) -> int:
        """Return the number of domain cells per side for the full bounding box, including the part outside the grid domain."""
        if self.cells[4] == 0:
            # Grid volume for this group is zero, so we can consider it as having zero cells per side.
            return 0
        else:
            # cell_max_indexes are inclusive, so we need to add 1 to get the count of cells along each axis.
            return int(self.cells[1][0] - self.cells[0][0] + 1)


def find_enclosing_sphere(centers: np.ndarray, radii: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> Tuple[np.ndarray, float]:
    """Approximate the minimum enclosing sphere of spheres.

    The objective is:
        minimize_c max_i ||c - x_i|| + r_i

    Parameters
    ----------
    centers : np.ndarray
        Sphere centers, shape `(N, 3)`.
    radii : np.ndarray
        Sphere radii, shape `(N,)`.
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance on center displacement.

    Returns
    -------
    Tuple[np.ndarray, float]
        Estimated enclosing sphere center and radius.
    """
    if len(centers) == 0:
        return np.zeros(3), 0.0
    if len(centers) == 1:
        return centers[0].copy(), float(radii[0])

    # Compute the initial guess as the mean of the centers. If all spheres have the same radius,
    # this is already the optimal solution. Otherwise, we will iteratively move towards the farthest sphere.
    c = centers.mean(axis=0)

    for k in range(max_iter):

        # Find the sphere that is farthest from the current center in terms of c2ray distance (center-to-center + radius).
        d = np.linalg.norm(centers - c[None, :], axis=1) + radii
        j = np.argmax(d)
        direction = centers[j] - c
        norm = np.linalg.norm(direction)
        if norm > 0.0:
            direction = direction / norm
        else:
            direction = np.zeros(3)

        # Move the center towards the farthest sphere by a fraction of the distance.
        eta = 1.0 / (k + 2.0)
        c_new = c + eta * direction * max(1e-12, norm)

        # Check for convergence. If the center displacement is smaller than the tolerance, we consider it converged.
        if np.linalg.norm(c_new - c) < tol:
            c = c_new
            break
        c = c_new

    R = np.max(np.linalg.norm(centers - c[None, :], axis=1) + radii)
    return c, float(R)


def morton_like_key(p: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray, bits: int = 10) -> int:
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
    x = np.clip((p - domain_min) / np.maximum(domain_max - domain_min, 1e-12), 0.0, 1.0 - 1e-12)

    # Scale to integer by shifting by bits.
    grid = (x * (1 << bits)).astype(int)

    def split_by_3(v: int) -> int:
        out = 0
        for i in range(bits):
            out |= ((v >> i) & 1) << (3 * i)
        return out

    return split_by_3(grid[0]) | (split_by_3(grid[1]) << 1) | (split_by_3(grid[2]) << 2)


def evaluate_group(group_sources: List[Source], grid: Grid) -> Group:
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
    cells = grid.find_cells_in_box(bbox_min, bbox_max)

    # Basic cost evaluation: number of sources times local cell count
    # TODO CB: this is a very rough estimate. A more accurate cost model could be implemented.
    cost = len(group_sources) * cells[4]

    return Group(
        sources=list(group_sources),
        center=c,
        radius=R,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        cells=cells,
        cost=cost,
    )

# TODO CB: this can be generalized with different grouping strategies, e.g. space-filling curve grouping, clustering-based grouping, etc...
# This function could also be interfaced to external domain decomposition libraries or tools such as Cornerstone.
def build_groups(
    sources: List[Source],
    grid: Grid,
    r_group_max: float = -1.0,
    nsrc_max: int = 12,
    ncell_max: int = 80000,
    cost_max: float = 500000
) -> List[Group]:
    """Grouping of sources in Morton-like spatial order.

    Sources are sorted spatially, then accumulated into a running group until
    the radius constraint is violated.

    Parameters
    ----------
    sources : List[Source]
        List of sources to group.
    grid : Grid
        Grid used to estimate local cell counts for groups.
    r_group_max : float
        Maximum allowed group radius (enclosing sphere of source spheres).
    nsrc_max : int
        Maximum allowed number of sources in a group.
    ncell_max : int
        Maximum allowed number of local cells covered by the group bounding box.
    cost_max : float
        Maximum allowed computational cost for the group.

    Returns
    -------
    List[Group]     List of groups that satisfy the constraints.
    """

    if not sources:
        return []

    # Spatial ordering (here we are using the physical source position)
    ordered = sorted(sources, key=lambda s: morton_like_key(s.pos, grid.get_domain_min(), grid.get_domain_max()))

    groups: List[Group] = []
    current: List[Source] = []

    def valid(g: Group) -> bool:
        print(f"Evaluating group with {len(g.sources)} sources, radius {g.radius:.2f}, ncell {g.get_num_cells()}, cost {g.cost:.2f}")
        # TODO CB: for debugging only, find the correct validity condition.
        if r_group_max > 0.0:
            return (
                g.radius <= r_group_max
                and len(g.sources) <= nsrc_max
                # and g.ncell_local <= ncell_max
                # and g.cost <= cost_max
            )
        else:
            return (
                len(g.sources) <= nsrc_max
                # and g.ncell_local <= ncell_max
                # and g.cost <= cost_max
            )

    for s in ordered:
        if not current:
            current = [s]
            continue

        trial = current + [s]
        gtrial = evaluate_group(trial, grid)

        if valid(gtrial):
            current = trial
        else:
            groups.append(evaluate_group(current, grid))
            current = [s]

    if current:
        groups.append(evaluate_group(current, grid))

    return groups


def assign_groups_to_ranks(groups: List[Group], nranks: int):
    """
    Groups to ranks assignement according to cost.

    Parameters
    ----------
    groups : List[Group]
        List of groups to assign.
    nranks : int
        Number of ranks to distribute groups across.

    Returns
    -------

    Tuple[List[List[Group]], List[float]]
        A tuple of (rank_groups, rank_costs), where rank_groups is a list of lists of groups assigned to each rank,
        and rank_costs is the total cost for each rank.
    """
    rank_groups = [[] for _ in range(nranks)]
    rank_costs = [0.0 for _ in range(nranks)]

    # TODO CB: this is a simple assignment. More sophisticated algorithms could be used for better load balancing
    for g in sorted(groups, key=lambda x: x.cost, reverse=True):
        r = int(np.argmin(rank_costs))
        rank_groups[r].append(g)
        rank_costs[r] += g.cost

    return rank_groups, rank_costs
