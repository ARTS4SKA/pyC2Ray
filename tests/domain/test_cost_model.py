import math
from collections.abc import Callable

import numpy as np
import pytest

from pyc2ray.asora_core import max_q_shells
from pyc2ray.domain.cost_model import NUM_COLUMN_DENSITY_BANKS, ASORACostModel
from pyc2ray.load_extensions import libasora

if libasora is None:
    pytest.skip("libasora.so missing, skipping tests", allow_module_level=True)

try:
    from pyc2ray.lib import libasoratest
except ImportError:
    libasoratest = None  # type:ignore

# The cost model mirrors ASORA's column density buffer layout, so check it against
# the C++ implementation.
requires_libasoratest = pytest.mark.skipif(
    libasoratest is None, reason="libasoratest missing"
)


@requires_libasoratest
def test_num_column_density_banks_matches_asora() -> None:
    assert NUM_COLUMN_DENSITY_BANKS == libasoratest.num_banks


@requires_libasoratest
@pytest.mark.parametrize("n_cells_per_side", [1, 10, 100])
def test_shell_counts_match_asora(n_cells_per_side: int) -> None:
    cost_model = ASORACostModel(
        available_memory=1,
        max_resident_blocks=1,
        is_periodic_mode_active=True,
    )
    # R spans both the radius-limited and the grid-limited (q_max capped) regimes
    for R in np.linspace(0.0, 2.0 * n_cells_per_side, 50):
        q_max = max_q_shells(R, n_cells_per_side)
        assert cost_model._compute_cells_in_shell(
            R, n_cells_per_side
        ) == libasoratest.cells_in_shell(q_max)
        assert cost_model._compute_cells_to_shell(
            R, n_cells_per_side
        ) == libasoratest.cells_to_shell(q_max)


R = 2.0
N_CELLS_PER_SIDE = 8
N_SRC = 10
MAX_RESIDENT_BLOCKS = 4

ModelFactory = Callable[..., ASORACostModel]


@pytest.fixture
def make_model() -> ModelFactory:
    def _make(available_memory: int, budget: float = 1.0) -> ASORACostModel:
        return ASORACostModel(
            available_memory=available_memory,
            max_resident_blocks=MAX_RESIDENT_BLOCKS,
            is_periodic_mode_active=True,
            budget=budget,
        )

    return _make


@pytest.fixture
def footprints(make_model: ModelFactory) -> tuple[int, int]:
    """Bytes of the local fields and sources, and of one column density ring buffer."""
    cells_in_shell = make_model(1)._compute_cells_in_shell(R, N_CELLS_PER_SIDE)
    baseline = 8 * 3 * N_CELLS_PER_SIDE**3 + 20 * N_SRC
    return baseline, 8 * NUM_COLUMN_DENSITY_BANKS * cells_in_shell


def test_group_at_full_concurrency(
    make_model: ModelFactory, footprints: tuple[int, int]
) -> None:
    """With enough memory, every resident block gets a source and a buffer."""
    baseline, buffer = footprints
    full_concurrency = baseline + MAX_RESIDENT_BLOCKS * buffer
    model = make_model(10 * full_concurrency)

    mem_cost, comp_cost = model.compute_group_costs(R, N_CELLS_PER_SIDE, N_SRC)

    assert mem_cost == full_concurrency
    assert comp_cost == model._compute_cells_to_shell(R, N_CELLS_PER_SIDE) * math.ceil(
        N_SRC / MAX_RESIDENT_BLOCKS
    )


def test_group_below_full_concurrency_uses_the_blocks_that_fit(
    make_model: ModelFactory, footprints: tuple[int, int]
) -> None:
    """When only part of the resident blocks get a buffer, the group runs with those."""
    baseline, buffer = footprints
    model = make_model(baseline + 2 * buffer)

    mem_cost, comp_cost = model.compute_group_costs(R, N_CELLS_PER_SIDE, N_SRC)

    assert mem_cost == baseline + 2 * buffer
    assert mem_cost <= model.max_memory_cost_per_group
    assert comp_cost == model._compute_cells_to_shell(R, N_CELLS_PER_SIDE) * math.ceil(
        N_SRC / 2
    )


def test_budget_limits_the_memory_cost(
    make_model: ModelFactory, footprints: tuple[int, int]
) -> None:
    """Only a fraction of the available memory is used for the group."""
    baseline, buffer = footprints
    available = baseline + MAX_RESIDENT_BLOCKS * buffer
    model = make_model(available, budget=0.9)
    fitting_blocks = math.floor((0.9 * available - baseline) / buffer)

    mem_cost, _ = model.compute_group_costs(R, N_CELLS_PER_SIDE, N_SRC)

    # All the buffers fit in the available memory, but not in the budget.
    assert fitting_blocks < MAX_RESIDENT_BLOCKS
    assert mem_cost == baseline + fitting_blocks * buffer
    assert mem_cost <= 0.9 * available


def test_group_that_does_not_fit_is_rejected_without_raising(
    make_model: ModelFactory, footprints: tuple[int, int]
) -> None:
    """A group without room for a single buffer is reported, not raised.

    The grouping evaluates trial merges, so a merge that does not fit has to come back
    with a memory cost above the cap for the grouping to reject it.
    """
    baseline, buffer = footprints
    model = make_model(baseline + buffer - 1)

    mem_cost, _ = model.compute_group_costs(R, N_CELLS_PER_SIDE, N_SRC)

    assert mem_cost > model.max_memory_cost_per_group
