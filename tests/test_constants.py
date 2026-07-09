import pytest

import pyc2ray.constants as c


def test_constants():
    assert pytest.approx(c.year2s) == 31557600.0
    assert pytest.approx(c.ev2hz) == 241798924208491.78
    assert pytest.approx(c.ev2k) == 11604.518121550082
    assert pytest.approx(c.pc) == 3.0856775814913674e18
    assert pytest.approx(c.kpc) == 3.0856775814913673e21
    assert pytest.approx(c.Mpc) == 3.0856775814913676e24
    assert pytest.approx(c.M_sun) == 1.988409870698051e33
    assert pytest.approx(c.R_sun) == 69570000000.0
    assert pytest.approx(c.m_p) == 1.67262192369e-24
    assert pytest.approx(c.k_B) == 1.380649e-16
    assert pytest.approx(c.h) == 6.62607015e-27
    assert pytest.approx(c.c) == 29979245800.0
    assert pytest.approx(c.pi) == 3.141592653589793
