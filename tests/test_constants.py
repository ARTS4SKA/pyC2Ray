import pytest

import pyc2ray.constants as c


def test_constants():
    assert c.year2s == pytest.approx(31557600.0)
    assert c.ev2hz == pytest.approx(241798924208491.78)
    assert c.ev2k == pytest.approx(11604.518121550082)
    assert c.hz2k == pytest.approx(4.799243073366221e-11)
    assert c.pc == pytest.approx(3.0856775814913674e18)
    assert c.kpc == pytest.approx(3.0856775814913673e21)
    assert c.Mpc == pytest.approx(3.0856775814913676e24)
    assert c.msun2g == pytest.approx(1.988409870698051e33)
    assert c.m_p == pytest.approx(1.67262192369e-24)
