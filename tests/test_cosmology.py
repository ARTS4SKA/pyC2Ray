from dataclasses import FrozenInstanceError

import astropy.constants as cst
import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy.tests.helper import assert_quantity_allclose

from pyc2ray.cosmology import Cosmology21cm

ABU_H = 0.926
ABU_HE = 0.074
ABU_HE_MASS = 0.2486
COSMO_KW = {
    "H0": Planck18.H0,
    "Om0": Planck18.Om0,
    "Ob0": Planck18.Ob0,
    "Tcmb0": Planck18.Tcmb0,
}


ABU_PARAMS = ["abu_h", "abu_he", "abu_he_mass"]


@pytest.fixture
def cosmo():
    return Cosmology21cm(
        **COSMO_KW,
        abu_h=ABU_H,
        abu_he=ABU_HE,
        abu_he_mass=ABU_HE_MASS,
        name="Cosmology21cm",  # NOTE: Needed for clone, bug in astropy
    )


def test_class_constants():
    assert_quantity_allclose(Cosmology21cm.nu0, 1420.405751768 * u.MHz)
    assert_quantity_allclose(Cosmology21cm.lambda0, 21.106 * u.cm, rtol=1e-4)
    assert_quantity_allclose(Cosmology21cm.A10, 2.85e-15 / u.s)


def test_init(cosmo):
    assert isinstance(cosmo, FlatLambdaCDM)
    assert_quantity_allclose(cosmo.H0, Planck18.H0)
    assert cosmo.Om0 == pytest.approx(Planck18.Om0)
    assert cosmo.Ob0 == pytest.approx(Planck18.Ob0)

    assert cosmo.abu_h == ABU_H
    assert cosmo.abu_he == ABU_HE
    assert cosmo.abu_he_mass == ABU_HE_MASS
    assert cosmo.abu_h_mass == pytest.approx(1.0 - ABU_HE_MASS)
    assert cosmo.mean_molecular == pytest.approx(ABU_H + 4.0 * ABU_HE)
    assert cosmo.fHc == pytest.approx(
        (1.0 - ABU_HE_MASS) * COSMO_KW["Ob0"] / COSMO_KW["Om0"]
    )


def test_defaults():
    cosmo = Cosmology21cm(**COSMO_KW)
    assert cosmo.abu_h == pytest.approx(0.926)
    assert cosmo.abu_he == pytest.approx(0.074)
    assert cosmo.abu_he_mass == pytest.approx(0.2486)


@pytest.mark.parametrize("name", ABU_PARAMS)
def test_abundances_are_parameters(cosmo, name):
    assert name in Cosmology21cm.parameters
    assert name in cosmo.parameters
    assert cosmo.parameters[name] == getattr(cosmo, name)
    assert name in repr(cosmo)


def test_class_constants_are_not_parameters(cosmo):
    for name in ("nu0", "lambda0", "A10"):
        assert name not in cosmo.parameters
    with pytest.raises(TypeError):
        Cosmology21cm(**COSMO_KW, nu0=1.0 * u.Hz)


@pytest.mark.parametrize("name", ABU_PARAMS)
def test_frozen(cosmo, name):
    with pytest.raises(FrozenInstanceError):
        setattr(cosmo, name, 0.5)


def test_clone_keeps_abundances(cosmo):
    other = cosmo.clone(Om0=0.3)
    assert isinstance(other, Cosmology21cm)
    assert other.Om0 == pytest.approx(0.3)
    for name in ABU_PARAMS:
        assert getattr(other, name) == getattr(cosmo, name)


def test_clone_updates_derived(cosmo):
    other = cosmo.clone(abu_he_mass=0.25)
    assert other.abu_he_mass == pytest.approx(0.25)
    assert other.abu_h_mass == pytest.approx(0.75)
    assert other.fHc == pytest.approx(0.75 * other.Ob0 / other.Om0)
    assert other.nH != cosmo.nH


def test_equality(cosmo):
    same = Cosmology21cm(
        **COSMO_KW,
        abu_h=ABU_H,
        abu_he=ABU_HE,
        abu_he_mass=ABU_HE_MASS,
        name="Cosmology21cm",
    )
    assert cosmo == same
    assert cosmo != cosmo.clone(abu_he=0.08)
    assert cosmo != FlatLambdaCDM(**COSMO_KW)


@pytest.mark.parametrize("name", ABU_PARAMS)
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_validator_out_of_range(name, value):
    with pytest.raises(ValueError, match=name):
        Cosmology21cm(**COSMO_KW, **{name: value})


@pytest.mark.parametrize("name", ABU_PARAMS)
def test_validator_casts_to_float(name):
    cosmo = Cosmology21cm(**COSMO_KW, **{name: 0.5 * u.one})
    assert isinstance(getattr(cosmo, name), float)


def test_nH(cosmo):
    expected = (
        (1 - ABU_HE_MASS)
        * Planck18.Ob0
        * Planck18.critical_density0
        / (cosmo.mean_molecular * cst.m_p)
    )
    assert cosmo.nH.unit == u.cm**-3
    assert_quantity_allclose(cosmo.nH, expected)


@pytest.mark.parametrize("z", [0.0, 6.0, 9.0, 20.0])
def test_mean_nH(cosmo, z):
    assert_quantity_allclose(cosmo.mean_nH(z), cosmo.nH * (1 + z) ** 3)


@pytest.mark.parametrize("z", [0.0, 6.0, 8.0, 9.0, 10.0])
def test_z2nu(cosmo, z: float):
    assert cosmo.z2nu(z).unit == u.MHz
    assert_quantity_allclose(cosmo.z2nu(z), 1420.405751768 * u.MHz / (1 + z))


def test_Tstar(cosmo):
    assert cosmo.Tstar.unit == u.K
    assert_quantity_allclose(cosmo.Tstar, 68.168 * u.mK, rtol=1e-4)


def test_meanT(cosmo):
    expected = (
        3
        * Cosmology21cm.lambda0**3
        * Cosmology21cm.A10
        * cosmo.Tstar
        * cosmo.nH
        / (32 * np.pi * cosmo.H0)
    )
    assert cosmo.meanT.unit == u.mK
    assert_quantity_allclose(cosmo.meanT, expected)


@pytest.mark.parametrize("z", [7.0, 9.0, 12.0])
def test_dTb_mean_neutral(cosmo, z):
    expected = cosmo.meanT * (1 + z) ** 2 / cosmo.efunc(z)
    assert cosmo.dTb(z).unit == u.mK
    assert_quantity_allclose(cosmo.dTb(z), expected)


def test_dTb_ionized_fraction(cosmo):
    z = 8.0
    dT0 = cosmo.dTb(z)
    assert_quantity_allclose(cosmo.dTb(z, xHII=np.array([1.0])), 0.0 * u.K)
    assert_quantity_allclose(cosmo.dTb(z, xHII=np.array([0.5])), 0.5 * dT0)


def test_dTb_array(cosmo):
    z = 8.0
    rng = np.random.default_rng(42)
    ndens = rng.uniform(0.5, 1.5, size=(4, 4, 4)) * cosmo.mean_nH(z)
    xHII = rng.uniform(0.0, 1.0, size=ndens.shape)

    dT = cosmo.dTb(z, ndens=ndens, xHII=xHII)
    assert dT.shape == ndens.shape
    expected = cosmo.dTb(z) * (1 - xHII) * ndens / ndens.mean()
    assert_quantity_allclose(dT, expected)


def test_dTb_ndens_plain_array(cosmo):
    z = 8.0
    ndens = np.array([1.0, 2.0, 3.0]) * cosmo.mean_nH(z)
    assert_quantity_allclose(cosmo.dTb(z, ndens=ndens.value), cosmo.dTb(z, ndens=ndens))


def test_dTb_spin_temperature(cosmo):
    z = 8.0
    Tcmb = cosmo.Tcmb(z)
    dT0 = cosmo.dTb(z)
    assert_quantity_allclose(cosmo.dTb(z, Ts=Tcmb), 0.0 * u.K, atol=1e-10 * u.K)
    assert_quantity_allclose(cosmo.dTb(z, Ts=2 * Tcmb), 0.5 * dT0)
    assert_quantity_allclose(cosmo.dTb(z, Ts=0.5 * Tcmb), -dT0)
    # plain float is interpreted as K
    assert_quantity_allclose(cosmo.dTb(z, Ts=2 * Tcmb.value), 0.5 * dT0)


def test_dTb_helium_not_implemented(cosmo):
    with pytest.raises(NotImplementedError):
        cosmo.dTb(8.0, flag_he=True)
