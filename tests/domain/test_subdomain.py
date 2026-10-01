"""Unit tests for domain decomposition subdomain class."""

from __future__ import annotations

import numpy as np
import pytest

from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import Source, SourceGroup
from pyc2ray.domain.subdomain import Subdomain


def _subdomain(sources: list[Source]) -> Subdomain:
    group = SourceGroup(
        id=0,
        sources=sources,
        center=np.array([1.0, 1.0, 1.0]),
        radius=1.0,
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([2.0, 2.0, 2.0]),
        mem_cost=0.0,
        comp_cost=0.0,
    )
    grid = RegularGrid(
        cell_size=0.5, num_cells=4, offset=np.array([2, 2, 2], dtype=np.int64)
    )
    return Subdomain(group, grid)


def _sources() -> list[Source]:
    return [
        Source(id=0, pos=np.array([1.25, 1.75, 2.75]), strength=2.0, radius=0.1),
        Source(id=1, pos=np.array([1.0, 1.0, 1.0]), strength=3.0, radius=0.1),
    ]


def test_local_sources_positions_are_mapped_to_the_local_grid() -> None:
    subdomain = _subdomain(_sources())

    np.testing.assert_array_equal(
        subdomain.get_local_sources_positions(), np.array([[0, 1, 3], [0, 0, 0]])
    )
    np.testing.assert_array_equal(
        subdomain.get_local_sources_strengths(), np.array([2.0, 3.0])
    )


def test_local_sources_accessors_return_the_cached_array() -> None:
    subdomain = _subdomain(_sources())

    assert (
        subdomain.get_local_sources_positions()
        is subdomain.get_local_sources_positions()
    )
    assert (
        subdomain.get_local_sources_strengths()
        is subdomain.get_local_sources_strengths()
    )


@pytest.mark.parametrize("num_sources", [0, 2])
def test_local_sources_accessors_are_read_only(num_sources: int) -> None:
    """Writing to the cached arrays must fail rather than corrupt later calls."""
    subdomain = _subdomain(_sources()[:num_sources])

    positions = subdomain.get_local_sources_positions()
    strengths = subdomain.get_local_sources_strengths()

    assert positions.shape == (num_sources, 3)
    assert strengths.shape == (num_sources,)
    with pytest.raises(ValueError):
        positions[...] = 1
    with pytest.raises(ValueError):
        strengths[...] = 1.0
