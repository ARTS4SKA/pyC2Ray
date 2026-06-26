import pytest

import pyc2ray.constants as c


def test_constants():
    assert pytest.approx(c.year2s) == 31557600.0
    assert pytest.approx(c.ev2fr) == 241798924208491.78
    assert pytest.approx(c.ev2k) == 11604.518121550082
    assert pytest.approx(c.pc) == 3.0856775814913674e18
    assert pytest.approx(c.kpc) == 3.0856775814913673e21
    assert pytest.approx(c.Mpc) == 3.0856775814913676e24
    assert pytest.approx(c.msun2g) == 1.988409870698051e33
    assert pytest.approx(c.m_p) == 1.67262192369e-24
