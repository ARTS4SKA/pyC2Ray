import math

import numpy as np
import pytest

import pyc2ray.constants as c
from pyc2ray.radiation.blackbody import BlackBodySource, BlackBodySource_Multifreq

temp = 5e4
ion_freq_HI = c.ev2hz * 13.598
ion_freq_HeI = c.ev2hz * 24.587
ion_freq_HeII = c.ev2hz * 54.416


class TestBlackBodySource:
    def test_init(self):
        rad = BlackBodySource(
            temp,
            (ion_freq_HI, 10 * ion_freq_HI),
            ion_freq_HI,
            2.8,
        )
        assert rad.temp == temp
        assert not rad.grey
        assert rad.S_star_ref == 1e48
        assert pytest.approx(rad.ion_freq) == ion_freq_HI
        assert pytest.approx(rad.pl_index) == 2.8

        assert pytest.approx(math.sqrt(rad.R_star2)) == 112807093977.5918

        assert pytest.approx(rad.SED(ion_freq_HI)) == 3362672211.7988915
        assert pytest.approx(rad.SED(10 * ion_freq_HI)) == 0.1486676621134524
        assert pytest.approx(rad.SED(500 * ion_freq_HI)) == 0.0

    def test_integrands(self):
        rad = BlackBodySource(temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8)
        tau = np.logspace(-2, 4, num=7)

        # At ion_freq_HI thin and thick are the same
        thin = rad.make_photo_integrand(rad.integrand_thin, tau)
        thick = rad.make_photo_integrand(rad.integrand_thick, tau)

        # fmt: off
        assert np.allclose(
            thin(ion_freq_HI),
            [
                5.32383120e32, 4.86561536e32, 1.97821158e32,
                2.44130704e28, 2.00041007e-11, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick(ion_freq_HI),
            [
                5.32383120e32, 4.86561536e32, 1.97821158e32,
                2.44130704e28, 2.00041007e-11, 0.0, 0.0,
            ],
        )
        # fmt: on

        # At a frequency > ion_freq_HI, differences appear
        thin = rad.make_photo_integrand(rad.integrand_thin, tau)
        thick = rad.make_photo_integrand(rad.integrand_thick, tau)

        # fmt: off
        assert np.allclose(
            thin(ion_freq_HeI),
            [
                2.50526817e31, 2.46269511e31, 2.07479373e31,
                3.73781412e30, 1.34600119e23, 4.93540609e-52, 0.0,
            ],
        )
        assert np.allclose(
            thick(ion_freq_HeI),
            [
                1.31552660e32, 1.29317131e32, 1.08948270e32,
                1.96274153e31, 7.06790751e23, 2.59160199e-51, 0.0,
            ],
        )
        # fmt: on

    def test_make_photo_tables(self):
        rad = BlackBodySource(temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8)
        tau = np.logspace(-2, 4, num=7)
        thin, thick = rad.make_photo_tables(tau)

        # fmt: off
        assert np.allclose(
            thin,
            [
                4.52245560e47, 4.28275587e47, 2.55247190e47,
                1.21275658e46, 1.77928961e43, 1.33762254e38, 8.20157669e27,
            ],
        )
        assert np.allclose(
            thick,
            [
                9.95463769e47, 9.55851708e47, 6.56184687e47,
                8.44510725e46, 5.81381324e44, 2.19422652e40, 4.64344011e30,
            ],
        )
        # fmt: on

    def test_make_heat_tables(self):
        rad = BlackBodySource(temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8)
        tau = np.logspace(-2, 4, num=7)
        thin, thick = rad.make_heat_tables(tau)

        # fmt: off
        assert np.allclose(
            thin,
            [
                2.63943454e36, 2.55073470e36, 1.84686340e36,
                2.40081454e35, 9.26272363e32, 1.48653107e28, 1.53863900e18,
            ],
        )
        assert np.allclose(
            thick,
            [
                1.06710711e37, 1.04375432e37, 8.48112762e36,
                2.24607370e36, 3.37646749e34, 2.56077161e30, 8.74008278e20,
            ],
        )
        # fmt: on

    def test_init_grey(self):
        rad = BlackBodySource(
            temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8, grey=True
        )

        assert rad.temp == temp
        assert rad.grey
        assert rad.S_star_ref == 1e48
        assert pytest.approx(rad.ion_freq) == ion_freq_HI
        assert pytest.approx(rad.pl_index) == 2.8

        assert pytest.approx(math.sqrt(rad.R_star2)) == 112807093977.5918

    def test_integrands_grey(self):
        rad = BlackBodySource(
            temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8, grey=True
        )
        tau = np.logspace(-2, 4, num=7)

        # When grey, i.e. no frequency dependency, thin and thick are always identical
        thin = rad.make_photo_integrand(rad.integrand_thin, tau)
        thick = rad.make_photo_integrand(rad.integrand_thick, tau)

        # fmt: off
        assert np.allclose(
            thin(ion_freq_HI),
            [
                5.32383120e32, 4.86561536e32, 1.97821158e32,
                2.44130704e28, 2.00041007e-11, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick(ion_freq_HI),
            [
                5.32383120e32, 4.86561536e32, 1.97821158e32,
                2.44130704e28, 2.00041007e-11, 0.00000000e00, 0.0,
            ],
        )
        # fmt: on

        # Even at different frequencies
        thin = rad.make_photo_integrand(rad.integrand_thin, tau)
        thick = rad.make_photo_integrand(rad.integrand_thick, tau)

        # fmt: off
        assert np.allclose(
            thin(ion_freq_HeI),
            [
                1.30491959e32, 1.19260671e32, 4.84877705e31,
                5.98386625e27, 4.90318756e-12, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick(ion_freq_HeI),
            [
                1.30491959e32, 1.19260671e32, 4.84877705e31,
                5.98386625e27, 4.90318756e-12, 0.0, 0.0,
            ],
        )
        # fmt: on

    def test_make_photo_tables_grey(self):
        rad = BlackBodySource(
            temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8, grey=True
        )
        tau = np.logspace(-2, 4, num=7)

        thin, thick = rad.make_photo_tables(tau)

        # fmt: off
        assert np.allclose(
            thin,
            [
                9.90049834e47, 9.04837418e47, 3.67879441e47,
                4.53999298e43, 3.72007598e04, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick,
            [
                9.90049834e47, 9.04837418e47, 3.67879441e47,
                4.53999298e43, 3.72007598e04, 0.0, 0.0,
            ],
        )
        # fmt: on

    def test_make_heat_tables_grey(self):
        rad = BlackBodySource(
            temp, (ion_freq_HI, 10 * ion_freq_HI), ion_freq_HI, 2.8, grey=True
        )
        tau = np.logspace(-2, 4, num=7)

        thin, thick = rad.make_heat_tables(tau)

        # fmt: off
        assert np.allclose(
            thin,
            [
                1.05910739e+37, 9.67951271e+36, 3.93539619e+36,
                4.85666473e+32, 3.97955721e-07, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick,
            [
                1.05910739e+37, 9.67951271e+36, 3.93539619e+36,
                4.85666473e+32, 3.97955721e-07, 0.0, 0.0,
            ],
        )
        # fmt: on


class TestBlackBodySourceMultifreq:
    def test_init(self):
        rad = BlackBodySource_Multifreq(temp)
        assert rad.temp == temp
        assert not rad.grey
        assert rad.S_star_ref == 1e48
        assert pytest.approx(rad.ion_freq_HI) == ion_freq_HI
        assert pytest.approx(rad.ion_freq_HeI) == ion_freq_HeI
        assert pytest.approx(rad.ion_freq_HeII) == ion_freq_HeII

        assert rad.freqs.shape == (47, 2)

        assert rad.pl_index_HI.shape == (47,)
        assert rad.pl_index_HeI.shape == (47,)
        assert rad.pl_index_HeII.shape == (47,)

        assert pytest.approx(math.sqrt(rad.R_star2)) == 112807093976.10341

    def test_integrands(self):
        rad = BlackBodySource_Multifreq(temp)
        tau = np.logspace(-2, 4, num=7)

        pl = 2.8
        fmin = rad.freqs[20, 0]  # Reference frequency for HeI
        thin = rad.make_photo_integrand(rad.integrand_thin, tau, fmin, pl)
        thick = rad.make_photo_integrand(rad.integrand_thick, tau, fmin, pl)

        # At the reference frequency, thin and thick are the same
        # fmt: off
        assert np.allclose(
            thin(fmin),
            [
                3.48313895e30, 3.18334931e30, 1.29425325e30,
                1.59723539e26, 1.30877670e-13, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick(fmin),
            [
                3.48313895e30, 3.18334931e30, 1.29425325e30,
                1.59723539e26, 1.30877670e-13, 0.0, 0.0,
            ],
        )
        # fmt: on

        # At a different frequency, differences appear
        # fmt: off
        assert np.allclose(
            thin(2 * fmin),
            [
                5.24895546e25, 5.18156037e25, 4.55341579e25,
                1.25056072e25, 3.05337058e19, 2.29893589e-37, 0.0,
            ],
        )
        assert np.allclose(
            thick(2 * fmin),
            [
                3.65558491e26, 3.60864824e26, 3.17118295e26,
                8.70941071e25, 2.12649079e20, 1.60107195e-36, 0.0,
            ],
        )
        # fmt: on

    def test_make_photo_tables(self):
        rad = BlackBodySource_Multifreq(temp)
        tau = np.logspace(-2, 4, num=7)
        thin, thick = rad.make_photo_tables(tau)

        assert thin.shape == (47, len(tau))
        assert thick.shape == (47, len(tau))

        # Checking only the sum over the frequency dimension for each tau instead of the full table
        # fmt: off
        assert np.allclose(
            thin.sum(0),
            [
                6.184647024250836e47, 5.7891460351849175e47, 3.072508334893663e47,
                5.516476640292399e45, 1.0227665367725258e37, 6.548585449293689e-41, 0.0,
            ],
        )
        # fmt: on

        # fmt: off
        assert np.allclose(
            thick.sum(0),
            [
                9.937925025247458e47, 9.39932181625429e47, 5.5559530132293635e47,
                1.9292219639272893e46, 4.992707157142024e37, 3.3430213921598803e-40, 0.0,
            ],
        )
        # fmt: on

    def test_make_heat_tables(self):
        rad = BlackBodySource_Multifreq(temp)
        tau = np.logspace(-2, 4, num=7)
        thin, thick = rad.make_heat_tables(tau)

        assert thin.shape == (47, 3, len(tau))
        assert thick.shape == (47, 3, len(tau))

        # fmt: off
        assert np.allclose(
            thin.sum(0),
            [
                [
                    7.099579434282554e36, 6.59814234298793e36, 3.277880291114221e36,
                    6.740666550185626e34, 1.7275746439797953e26, 1.1481817221128119e-51, 0.0,
                ],
                [
                    1.7084431045943634e36, 1.565847273178549e36, 6.5512802620994465e35,
                    1.1042519719467497e32, 150166.46664930639, 2.1819731634006836e-153, 0.0,
                ],
                [
                    5.452878601784309e33, 5.029605514682601e33, 2.2474946719277343e33,
                    9.806960124943654e29, 84675.26519973791, 1.9991328327216495e-153, 0.0,
                ],
            ],
        )
        assert np.allclose(
            thick.sum(0),
            [
                [
                    1.0626229314386157e37, 1.0010135247798203e37, 5.760757233024756e36,
                    2.5429211679245207e35, 8.448855912889166e26, 5.861536052124382e-51, 0.0,
                ],
                [
                    1.7650542323644733e36, 1.617804408159352e36, 6.771712191516827e35,
                    1.1495135789202286e32, 254351.7510698966, 6.075242684844919e-153, 0.0,
                ],
                [
                    6.117844753335088e33, 5.646391189868075e33, 2.5391632063204517e33,
                    1.2129479628687637e30, 143702.930028088, 5.566166843186363e-153, 0.0,
                ],
            ],
        )
        # fmt: on

    def test_init_grey(self):
        rad = BlackBodySource_Multifreq(temp, grey=True)

        assert rad.temp == temp
        assert rad.grey
        assert rad.S_star_ref == 1e48
        assert pytest.approx(rad.ion_freq_HI) == ion_freq_HI
        assert pytest.approx(rad.ion_freq_HeI) == ion_freq_HeI
        assert pytest.approx(rad.ion_freq_HeII) == ion_freq_HeII

        assert pytest.approx(math.sqrt(rad.R_star2)) == 112807093976.10341

    def test_integrands_grey(self):
        rad = BlackBodySource_Multifreq(temp, grey=True)
        tau = np.logspace(-2, 4, num=7)

        # When grey, i.e. no frequency dependency, thin and thick are always identical
        pl = 2.8
        fmin = rad.freqs[20, 0]  # Reference frequency for HeI
        thin = rad.make_photo_integrand(rad.integrand_thin, tau, fmin, pl)
        thick = rad.make_photo_integrand(rad.integrand_thick, tau, fmin, pl)

        # fmt: off
        assert np.allclose(
            thin(fmin),
            [
                3.48313895e30, 3.18334931e30, 1.29425325e30,
                1.59723539e26, 1.30877670e-13, 0.0, 0.0,
            ],
        )
        assert np.allclose(
            thick(fmin),
            [
                3.48313895e30, 3.18334931e30, 1.29425325e30,
                1.59723539e26, 1.30877670e-13, 0.0, 0.0,
            ],
        )
        # fmt: on

        # fmt: off
        assert np.allclose(
            thin(2 * fmin),
            [
                3.62441169e+26, 3.31246287e+26, 1.34674690e+26, 1.66201771e+22,
                1.36185941e-17, 0.0, 0.0, 
            ],
        )
        assert np.allclose(
            thick(2 * fmin),
            [
                3.62441169e+26, 3.31246287e+26, 1.34674690e+26, 1.66201771e+22,
                1.36185941e-17, 0.0, 0.0, 
            ],
        )
        # fmt: on

    def test_make_photo_tables_grey(self):
        rad = BlackBodySource_Multifreq(temp, grey=True)
        tau = np.logspace(-2, 4, num=7)

        thin, thick = rad.make_photo_tables(tau)

        assert thin.shape == (47, len(tau))
        assert thick.shape == (47, len(tau))

        # fmt: off
        assert np.allclose(
            thin.sum(0),
            [
                9.90049834e47, 9.04837418e47, 3.67879441e47,
                4.53999298e43, 3.72007598e04, 0.0, 0.0,
            ],
        )

        assert np.allclose(
            thick.sum(0),
            [
                9.90049834e47, 9.04837418e47, 3.67879441e47,
                4.53999298e43, 3.72007598e04, 0.0, 0.0,
            ],
        )
        # fmt: on

    def test_make_heat_tables_grey(self):
        rad = BlackBodySource_Multifreq(temp, grey=True)
        tau = np.logspace(-2, 4, num=7)

        thin, thick = rad.make_heat_tables(tau)

        assert thin.shape == (47, 3, len(tau))
        assert thick.shape == (47, 3, len(tau))

        # fmt: off
        assert np.allclose(
            thin.sum(0),
            [
                [
                    1.05910739e37, 9.67951271e36, 3.93539619e36,
                    4.85666473e32, 3.97955721e-07, 0.0, 0.0,
                ],
                [
                    1.76448825e36, 1.61262084e36, 6.55642707e35,
                    8.09127380e31, 6.62999996e-08, 0.0, 0.0,
                ],
                [
                    6.11120054e33, 5.58521675e33, 2.27077967e33,
                    2.80236475e29, 2.29626121e-10, 0.0, 0.0,
                ],
            ],
        )
        assert np.allclose(
            thick.sum(0),
            [
                [
                    1.05910739e37, 9.67951271e36, 3.93539619e36,
                    4.85666473e32, 3.97955721e-07, 0.0, 0.0,
                ],
                [
                    1.76448825e36, 1.61262084e36, 6.55642707e35,
                    8.09127380e31, 6.62999996e-08, 0.0, 0.0,
                ],
                [
                    6.11120054e33, 5.58521675e33, 2.27077967e33,
                    2.80236475e29, 2.29626121e-10, 0.0, 0.0,
                ],
            ],
        )
        # fmt: on
