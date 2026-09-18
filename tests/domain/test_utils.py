"""Unit tests for domain-geometry utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from pyc2ray.domain.utils import (
    evaluate_sphere_intersection,
    find_enclosing_sphere,
)


def assert_contains_all_spheres(
    centers: np.ndarray, radii: np.ndarray, center: np.ndarray, radius: float
) -> None:
    distances = np.linalg.norm(centers - center[None, :], axis=1) + radii
    assert np.all(distances <= radius + 1e-10)


def test_find_enclosing_sphere_empty_singleton_and_two_sphere_containment() -> None:
    """The enclosing sphere must contain all input spheres, and the function must handle edge cases."""
    center_empty, radius_empty = find_enclosing_sphere(np.empty((0, 3)), np.empty((0,)))
    np.testing.assert_array_equal(center_empty, np.zeros(3))
    assert radius_empty == 0.0

    centers_single = np.array([[1.0, 2.0, 3.0]])
    radii_single = np.array([0.5])
    center_single, radius_single = find_enclosing_sphere(centers_single, radii_single)
    np.testing.assert_array_equal(center_single, centers_single[0])
    assert radius_single == pytest.approx(radii_single[0])

    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    center, radius = find_enclosing_sphere(centers, radii)

    assert_contains_all_spheres(centers, radii, center, radius)
    assert radius == pytest.approx(2.0, abs=1e-2)


def test_find_enclosing_sphere_contains_unequal_and_nested_spheres() -> None:
    """The enclosing sphere must contain all input spheres, even if they are nested or have unequal radii."""
    centers = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [3.0, -2.0, 1.0]], dtype=float
    )
    radii = np.array([2.0, 0.25, 1.5], dtype=float)

    center, radius = find_enclosing_sphere(centers, radii)

    assert_contains_all_spheres(centers, radii, center, radius)
    assert radius >= radii.max()


def test_find_enclosing_sphere_is_translation_invariant_and_preserves_inputs() -> None:
    """The enclosing sphere is translation-invariant, and the input arrays are not modified."""
    centers = np.array(
        [[-1.0, 2.0, 0.5], [2.0, -3.0, 1.0], [4.0, 1.5, -2.0]], dtype=float
    )
    radii = np.array([0.25, 1.5, 0.75], dtype=float)
    centers_before = centers.copy()
    radii_before = radii.copy()
    offset = np.array([12.5, -7.25, 3.0])

    center, radius = find_enclosing_sphere(centers, radii)
    translated_center, translated_radius = find_enclosing_sphere(
        centers + offset, radii
    )

    assert_contains_all_spheres(centers, radii, center, radius)
    assert_contains_all_spheres(
        centers + offset, radii, translated_center, translated_radius
    )
    np.testing.assert_allclose(translated_center, center + offset, atol=1e-10)
    assert translated_radius == pytest.approx(radius, abs=1e-10)
    np.testing.assert_array_equal(centers, centers_before)
    np.testing.assert_array_equal(radii, radii_before)


def test_evaluate_sphere_intersection_disjoint_overlapping_and_nested() -> None:
    origin = np.zeros(3)

    # A (2, 3, 6) offset puts the centers exactly 7.0 apart with every
    # coordinate contributing a different amount, so a dropped or duplicated
    # component changes the computed distance and fails the tight cases below.
    offset = np.array([2.0, 3.0, 6.0])
    assert np.linalg.norm(offset) == 7.0

    # Just disjoint: the radii fall a hair short of the separation.
    assert not evaluate_sphere_intersection(origin, 3.0, offset, 3.9)

    # Just overlapping: same geometry, radii nudged past the separation.
    assert evaluate_sphere_intersection(origin, 3.0, offset, 4.1)

    # A small sphere fully inside a large one still counts as intersecting.
    assert evaluate_sphere_intersection(origin, 9.0, offset, 0.25)

    # Concentric spheres.
    assert evaluate_sphere_intersection(origin, 2.0, origin.copy(), 0.5)


def test_evaluate_sphere_intersection_treats_tangency_as_disjoint() -> None:
    """Tangent spheres are reported as *not* intersecting.

    The comparison is strict, so spheres touching at exactly one point are
    excluded. This pins down current behavior rather than endorsing
    it: see the TODO in :func:`evaluate_sphere_intersection`, which notes that
    the grouping logic may want tangency to count as touching.

    The (2, 3, 6) offset is a Pythagorean quadruple, so the distance is
    exactly 7.0 in floating point: the assertion tests the comparison and not
    round-off, while still exercising all three coordinates.
    """
    center_a = np.zeros(3)
    center_b = np.array([2.0, 3.0, 6.0])
    assert np.linalg.norm(center_b - center_a) == 7.0

    assert not evaluate_sphere_intersection(center_a, 3.0, center_b, 4.0)

    # Growing either sphere by a small amount makes them overlap.
    assert evaluate_sphere_intersection(center_a, 3.0 + 1e-9, center_b, 4.0)
    assert evaluate_sphere_intersection(center_a, 3.0, center_b, 4.0 + 1e-9)


def test_evaluate_sphere_intersection_is_symmetric_and_preserves_inputs() -> None:
    center_a = np.array([-1.0, 2.0, 0.5])
    center_b = np.array([2.0, -3.0, 1.0])
    center_a_before = center_a.copy()
    center_b_before = center_b.copy()

    forward = evaluate_sphere_intersection(center_a, 2.0, center_b, 4.0)
    backward = evaluate_sphere_intersection(center_b, 4.0, center_a, 2.0)

    assert forward == backward
    # A numpy bool would propagate into the grouping predicates; keep it plain.
    assert isinstance(forward, bool)
    np.testing.assert_array_equal(center_a, center_a_before)
    np.testing.assert_array_equal(center_b, center_b_before)


def test_evaluate_sphere_intersection_handles_zero_radii_and_integer_centers() -> None:
    origin = np.zeros(3)

    # Degenerate point spheres: two coincident points do not intersect under the
    # strict comparison, but a point strictly inside a sphere does.
    assert not evaluate_sphere_intersection(origin, 0.0, origin.copy(), 0.0)
    assert evaluate_sphere_intersection(origin, 0.0, np.array([0.5, 0.0, 0.0]), 1.0)

    # Source positions are binned cell indices, so integer-dtype centers reach
    # these functions in practice.
    center_a = np.array([10, 20, 30])
    center_b = center_a + np.array([2, 3, 6])
    assert evaluate_sphere_intersection(center_a, 3.0, center_b, 4.1)
    assert not evaluate_sphere_intersection(center_a, 3.0, center_b, 3.9)
