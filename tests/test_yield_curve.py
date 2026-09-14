import numpy as np
from core.yield_curve import YieldCurveBootstrapper, NelsonSiegelSvensson


def test_one_year_deposit_zero():
    curve = YieldCurveBootstrapper()
    curve.add_cash_rate(1.0, 0.05)
    t, z = curve.get_curve()
    df = 1.0 / 1.05
    expected = -np.log(df) / 1.0
    assert abs(z[0] - expected) < 1e-12
    assert abs(curve.discount_factors[0] - df) < 1e-12


def test_two_year_swap_strip_is_finite():
    curve = YieldCurveBootstrapper()
    curve.add_cash_rate(1.0, 0.05)
    curve.add_swap_rate(2.0, 0.06)
    t, z = curve.get_curve()
    assert t.tolist() == [1.0, 2.0]
    assert np.all(np.isfinite(z))
    assert z[1] > 0


def test_nss_fits_flat_curve():
    t = np.array([1.0, 2.0, 5.0, 10.0])
    y = np.full_like(t, 0.05)
    nss = NelsonSiegelSvensson()
    nss.fit(t, y)
    assert nss.params is not None
    fitted = nss.get_curve(t)
    assert np.max(np.abs(fitted - y)) < 0.01
