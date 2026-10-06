import numpy as np
import pytest

from pyc2ray.domain.cost_model import NUM_COLUMN_DENSITY_BANKS, pyC2RayCostModel

# The cost model mirrors ASORA's column density buffer layout, so check it against
# the C++ implementation (skipped when ASORA is not built).
libasoratest = pytest.importorskip("pyc2ray.lib.libasoratest")


def test_num_column_density_banks_matches_asora() -> None:
    assert NUM_COLUMN_DENSITY_BANKS == libasoratest.num_banks


@pytest.mark.parametrize("n_cells_per_side", [1, 10, 100])
def test_shell_counts_match_asora(n_cells_per_side: int) -> None:
    cost_model = pyC2RayCostModel(
        max_memory_cost_per_group=1.0,
        source_batch_size=1,
        is_periodic_mode_active=True,
        photo_ion_table_size=1,
    )
    # R spans both the radius-limited and the grid-limited (q_max capped) regimes
    for R in np.linspace(0.0, 2.0 * n_cells_per_side, 50):
        q_max = cost_model._compute_q_max(R, n_cells_per_side)
        assert cost_model._compute_cells_in_shell(
            R, n_cells_per_side
        ) == libasoratest.cells_in_shell(q_max)
        assert cost_model._compute_cells_to_shell(
            R, n_cells_per_side
        ) == libasoratest.cells_to_shell(q_max)
