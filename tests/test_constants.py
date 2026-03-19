import astropy.constants as aconst
import astropy.units as u

from psoap import constants as C


def test_constants_are_astropy_backed():
    assert C.c_kms == aconst.c.to(u.km / u.s).value
    assert C.G == aconst.G.cgs.value
    assert C.M_sun == aconst.M_sun.cgs.value
    assert C.R_sun == aconst.R_sun.cgs.value


def test_day_and_solar_flux_consistency():
    assert C.day == u.day.to(u.s)
    expected_f_sun = C.L_sun / (4 * 3.141592653589793 * C.R_sun ** 2)
    assert C.F_sun == expected_f_sun
