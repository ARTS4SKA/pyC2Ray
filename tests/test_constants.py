import pyc2ray.constants as c


def test_constants():
    assert c.year2s == 31557600.0
    assert c.ev2fr == 241798924208491.78
    assert c.ev2k == 11604.518121550082
    assert c.pc == 3.0856775814913674e18
    assert c.kpc == 3.0856775814913673e21
    assert c.Mpc == 3.0856775814913676e24
    assert c.msun2g == 1.988409870698051e33
    assert c.m_p == 1.67262192369e-24
