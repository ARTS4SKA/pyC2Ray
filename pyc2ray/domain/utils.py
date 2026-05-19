import logging
import math
import numpy as np

from typing import Sequence, Tuple

from pyc2ray.domain.sources import SourceGroup

# TODO: this is probably not needed, check what is the strategy already adopted in pyC2Ray
def get_domain_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

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

def evaluate_sphere_intersection(center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float) -> bool:
    """
    Check if the two spheres intersect.

    Parameters
    ----------
    center_a : np.ndarray
        Center of the first sphere.
    radius_a : float
        Radius of the first sphere.
    center_b : np.ndarray
        Center of the second sphere.
    radius_b : float
        Radius of the second sphere.

    Returns
    -------
    bool    True if the two spheres intersect, False otherwise.
    """
    d = np.linalg.norm(center_a - center_b)
    if d < radius_a + radius_b:
        return True

    return False


logger = get_domain_logger(__name__)

def log_domain_decomposition_assignments(
    ranks_groups: Sequence[Sequence[SourceGroup]] | None,
    ranks_costs: Sequence[float],
    dr: float = 0.0,
) -> None:
    if ranks_groups is None:
        logger.info("No groups assigned to ranks.")
        return
    for rank, groups in enumerate(ranks_groups):
        n_local_sources = sum(group.get_num_sources() for group in groups)
        rank_cost = float(ranks_costs[rank])

        logger.info(
            "Scatter check | rank=%d groups=%d total num sources=%d",
            rank,
            len(groups),
            n_local_sources,
        )

        for group in groups:
            # TODO: refactoring with dr = 0 case.
            if dr > 0.0:
                logger.info(
                    (
                        "Local group index=%d cost=%.3e num sources=%d "
                        "computational_cost=%.3e "
                        "memory_cost=%.3e MB "
                        "center=(%.2f, %.2f, %.2f) "
                        "center in cell units=(%.2f, %.2f, %.2f) "
                        "radius=%.2f radius in cell units=(%.2f) "
                        "bounding_box_min=(%.2f, %.2f, %.2f) "
                        "bounding_box_max=(%.2f, %.2f, %.2f)"
                    ),
                    group.id,
                    rank_cost,
                    group.get_num_sources(),
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
                        "Local group index=%d cost=%.3e num sources=%d "
                        "computational_cost=%.3e "
                        "memory_cost=%.3e MB "
                        "center=(%.2f, %.2f, %.2f) "
                        "bounding_box_min=(%.2f, %.2f, %.2f) "
                        "bounding_box_max=(%.2f, %.2f, %.2f)"
                    ),
                    group.id,
                    rank_cost,
                    group.get_num_sources(),
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
