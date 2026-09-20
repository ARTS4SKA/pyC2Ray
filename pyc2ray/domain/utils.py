import logging
from collections.abc import Sequence
from math import sqrt
from pathlib import Path

import numpy as np

from pyc2ray.domain.sources import SourceGroup
from pyc2ray.utils.logutils import configure_logger


def find_enclosing_sphere(
    centers: np.ndarray, radii: np.ndarray, max_iter: int = 200, tol: float = 1e-8
) -> tuple[np.ndarray, float]:
    """Approximate the minimum enclosing sphere of spheres.

    The objective is:
        minimize_c max_i ||c - x_i|| + r_i

    Parameters
    ----------
    centers : Sphere centers, shape `(N, 3)`.
    radii : Sphere radii, shape `(N,)`.
    max_iter : Maximum number of fixed-point iterations.
    tol : Convergence tolerance on center displacement.

    Returns
    -------
    Estimated enclosing sphere center and radius.
    """
    if len(centers) == 0:
        return np.zeros(3), 0.0
    if len(centers) == 1:
        return centers[0].copy(), float(radii[0])

    # Compute the initial guess for the enclosing sphere center as the mean of the centers and
    # then iteratively move it towards the farthest sphere
    c = centers.mean(axis=0)
    tol2 = tol * tol
    n = centers.shape[0]
    all_radii_equal = bool(np.all(radii == radii[0]))
    r0 = float(radii[0])
    delta = np.empty((n, 3), dtype=float)
    dist2 = np.empty(n, dtype=float)
    d = np.empty(n, dtype=float)

    for k in range(max_iter):
        # Find the sphere that is farthest from the current center in terms of c2ray distance (center-to-center + radius).
        np.subtract(centers, c[None, :], out=delta)
        np.einsum("ij,ij->i", delta, delta, out=dist2)
        # TODO: this check could be moved outside the loop but we need to avoid code duplication.
        if all_radii_equal:
            j = int(np.argmax(dist2))
        else:
            np.sqrt(dist2, out=d)
            d += radii
            j = int(np.argmax(d))

        # Move the center towards the farthest sphere by a fraction of the distance.
        eta = 1.0 / (k + 2.0)
        step = eta * delta[j]
        c_new = c + step

        # Check for convergence. If the center displacement is smaller than the tolerance, we consider it converged.
        if np.dot(step, step) < tol2:
            c = c_new
            break
        c = c_new

    np.subtract(centers, c[None, :], out=delta)
    np.einsum("ij,ij->i", delta, delta, out=dist2)

    if all_radii_equal:
        R = np.sqrt(np.max(dist2)) + r0
    else:
        np.sqrt(dist2, out=d)
        d += radii
        R = np.max(d)
    return c, float(R)


def expand_enclosing_sphere(
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    sphere_x: float,
    sphere_y: float,
    sphere_z: float,
    sphere_radius: float,
) -> tuple[float, float, float, float]:
    """Grow a sphere just enough to also enclose a second sphere.

    The result always encloses both input spheres, but it is only an upper bound on the
    minimum enclosing sphere, and it depends on the order in which spheres are merged.
    This function should only be used as a fast approximation for preliminary group
    cost evaluation while the final enclosing sphere is computed with find_enclosing_sphere.

    Parameters
    ----------
    center_x, center_y, center_z : Center of the sphere to grow.
    radius : Radius of the sphere to grow.
    sphere_x, sphere_y, sphere_z : Center of the sphere to enclose.
    sphere_radius : Radius of the sphere to enclose.

    Returns
    -------
    Center coordinates and radius of a sphere enclosing both input spheres.
    """
    dx = sphere_x - center_x
    dy = sphere_y - center_y
    dz = sphere_z - center_z
    distance_squared = dx * dx + dy * dy + dz * dz

    # Concentric spheres: the larger one already encloses the smaller one.
    # Negative values are not possible here
    if distance_squared <= 0.0:
        return center_x, center_y, center_z, max(radius, sphere_radius)

    distance = sqrt(distance_squared)

    # One sphere already contains the other, so it is itself the enclosing sphere.
    if distance + sphere_radius <= radius:
        return center_x, center_y, center_z, radius
    if distance + radius <= sphere_radius:
        return sphere_x, sphere_y, sphere_z, sphere_radius

    # Otherwise the enclosing sphere is the one whose diameter spans the two far points of
    # the spheres along the line joining their centers.
    new_radius = 0.5 * (radius + distance + sphere_radius)
    shift = (new_radius - radius) / distance
    return (
        center_x + shift * dx,
        center_y + shift * dy,
        center_z + shift * dz,
        new_radius,
    )


def evaluate_sphere_intersection(
    center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float
) -> bool:
    """
    Check if the two spheres intersect.

    Parameters
    ----------
    center_a : Center of the first sphere.
    radius_a : Radius of the first sphere.
    center_b : Center of the second sphere.
    radius_b : Radius of the second sphere.

    Returns
    -------
    True if the two spheres intersect, False otherwise.
    """
    # TODO: Geometrically, tangency is typically considered intersection/touching.
    # In the grouping logic this can spuriously split groups for boundary cases.
    return bool(((center_a - center_b) ** 2).sum() < (radius_a + radius_b) ** 2)


logger = logging.getLogger(__name__)


def log_domain_decomposition_assignments(
    ranks_groups: Sequence[Sequence[SourceGroup]] | None,
    ranks_costs: Sequence[float],
    log_file: Path | None = None,
    dr: float = 0.0,
) -> None:
    if log_file is not None:
        configure_logger(log_file)

    if ranks_groups is None:
        logger.info("No groups assigned to ranks.")
        return
    for rank, groups in enumerate(ranks_groups):
        n_local_sources = sum(len(group) for group in groups)
        rank_cost = float(ranks_costs[rank])

        logger.info(
            "Scatter check | rank=%d groups=%d total num sources=%d total rank cost=%.3e",
            rank,
            len(groups),
            n_local_sources,
            rank_cost,
        )

        for group in groups:
            # TODO: refactoring with dr = 0 case.
            if dr > 0.0:
                logger.info(
                    (
                        "Local group index=%d num sources=%d "
                        "computational_cost=%.3e "
                        "memory_cost=%.3e MB "
                        "center=(%.2f, %.2f, %.2f) "
                        "center in cell units=(%.2f, %.2f, %.2f) "
                        "radius=%.2f radius in cell units=(%.2f) "
                        "bounding_box_min=(%.2f, %.2f, %.2f) "
                        "bounding_box_max=(%.2f, %.2f, %.2f)"
                    ),
                    group.id,
                    len(group),
                    group.comp_cost,
                    group.mem_cost / 1e6,
                    group.center[0],
                    group.center[1],
                    group.center[2],
                    group.center[0] / dr,
                    group.center[1] / dr,
                    group.center[2] / dr,
                    group.radius,
                    group.radius / dr,
                    group.bbox_min[0],
                    group.bbox_min[1],
                    group.bbox_min[2],
                    group.bbox_max[0],
                    group.bbox_max[1],
                    group.bbox_max[2],
                )
            else:
                logger.info(
                    (
                        "Local group index=%d num sources=%d "
                        "computational_cost=%.3e "
                        "memory_cost=%.3e MB "
                        "center=(%.2f, %.2f, %.2f) "
                        "bounding_box_min=(%.2f, %.2f, %.2f) "
                        "bounding_box_max=(%.2f, %.2f, %.2f)"
                    ),
                    group.id,
                    len(group),
                    group.comp_cost,
                    group.mem_cost / 1e6,
                    group.center[0],
                    group.center[1],
                    group.center[2],
                    group.bbox_min[0],
                    group.bbox_min[1],
                    group.bbox_min[2],
                    group.bbox_max[0],
                    group.bbox_max[1],
                    group.bbox_max[2],
                )
