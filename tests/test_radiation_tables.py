import numpy as np
import pytest

import pyc2ray.constants as c
from pyc2ray.radiation.radiation_tables import RadiationTables

ion_freq_HI = c.ev2hz * 13.598
ion_freq_HeI = c.ev2hz * 24.587
ion_freq_HeII = c.ev2hz * 54.416

NTOT = RadiationTables.NB1 + RadiationTables.NB2 + RadiationTables.NB3


@pytest.fixture(scope="module")
def radiation_tables() -> RadiationTables:
    return RadiationTables()


class TestRadiationTables:
    def test_freqs(self, radiation_tables: RadiationTables):
        rt = radiation_tables
        freq_min, freq_max = rt.freqs

        assert freq_min.shape == (NTOT,)
        assert freq_max.shape == (NTOT,)

        assert np.allclose(rt.freq_max[:-1], rt.freq_min[1:])
        assert (np.diff(rt.freq_min) > 0).all()
        assert (rt.freq_max > rt.freq_min).all()

    def test_cross_sections(self, radiation_tables: RadiationTables):
        rt = radiation_tables
        sigma_HI, sigma_HeI, sigma_HeII = rt.cross_sections

        assert sigma_HI.shape == (NTOT,)
        assert sigma_HeI.shape == (NTOT,)
        assert sigma_HeII.shape == (NTOT,)

        assert (rt.cross_section_HI >= 0.0).all()
        assert (rt.cross_section_HeI >= 0.0).all()
        assert (rt.cross_section_HeII >= 0.0).all()

    def test_powerlaw_indices(self, radiation_tables: RadiationTables):
        rt = radiation_tables
        pl_HI, pl_HeI, pl_HeII = rt.powerlaw_indices

        assert pl_HI.shape == (NTOT,)
        assert pl_HeI.shape == (NTOT,)
        assert pl_HeII.shape == (NTOT,)

    def test_factors(self, radiation_tables: RadiationTables):
        rt = radiation_tables
        factors = rt.factors

        assert factors.shape == (NTOT, 12)
        assert (factors >= 0.0).all()
        assert (factors <= 1.0).all()

    def test_ion_freqs(self, radiation_tables: RadiationTables):
        rt = radiation_tables

        assert pytest.approx(rt.ion_freq_HI) == ion_freq_HI
        assert pytest.approx(rt.ion_freq_HeI) == ion_freq_HeI
        assert pytest.approx(rt.ion_freq_HeII) == ion_freq_HeII

        assert rt.ion_freq_HI == rt.freq_min[0]
        assert rt.ion_freq_HeI == rt.freq_min[RadiationTables.NB1]
        assert (
            rt.ion_freq_HeII == rt.freq_min[RadiationTables.NB1 + RadiationTables.NB2]
        )
