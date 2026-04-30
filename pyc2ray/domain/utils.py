import logging
import numpy as np
from typing import Tuple

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
