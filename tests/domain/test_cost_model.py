from pyc2ray.domain.cost_model import pyC2RayCostModel


def test_num_column_density_banks_defaults_to_asora() -> None:
    from pyc2ray.asora_core import num_column_density_banks

    cost_model = pyC2RayCostModel(
        max_memory_cost_per_group=1.0,
        source_batch_size=1,
        is_periodic_mode_active=True,
        photo_ion_table_size=1,
    )
    assert cost_model.num_column_density_banks == num_column_density_banks()
