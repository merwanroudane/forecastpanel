import numpy as np

import forecastpanel as fp


def test_dm_h1_closed_form():
    rng = np.random.default_rng(0)
    d = rng.normal(0.2, 1, 100)
    r = fp.dm_test(d)
    assert np.isclose(r.statistic, d.mean() / np.sqrt(np.var(d, ddof=1) / 100))
    assert np.isclose(r.statistic_hln, np.sqrt((100 + 1 - 2) / 100) * r.statistic)


def test_dm_short_series_nan():
    assert np.isnan(fp.dm_test([1.0, 2.0, 3.0]).statistic)


def test_panel_dm_balanced_formula():
    rng = np.random.default_rng(1)
    D = rng.normal(-0.1, 1, size=(80, 25))
    r = fp.panel_dm_test(D)
    N = D.shape[1]
    stat = D.mean() / np.sqrt((np.var(D, axis=0, ddof=1) / N ** 2).sum() / 80)
    assert np.isclose(r.statistic, stat)
    assert np.isclose(fp.panel_dm_test(D, variance="ppp-legacy").statistic, stat)


def test_panel_dm_legacy_unbalanced_shrinks_by_sqrtN():
    rng = np.random.default_rng(2)
    D = rng.normal(-0.1, 1, size=(80, 25))
    D[0, 0] = np.nan
    a = fp.panel_dm_test(D).statistic
    b = fp.panel_dm_test(D, variance="ppp-legacy").statistic
    assert np.isclose(a / b, np.sqrt(25), rtol=0.02)


def test_panel_dm_csa_runs():
    rng = np.random.default_rng(3)
    D = rng.normal(0, 1, size=(60, 10))
    assert np.isfinite(fp.panel_dm_test(D, variance="cross-section-average").statistic)
