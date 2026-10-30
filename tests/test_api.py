import numpy as np
import pandas as pd
import pytest

import forecastpanel as fp


def _toy(N=8, T=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        y = 10 + rng.normal(size=T).cumsum()
        for t in range(T):
            rows.append({"firm": f"F{i}", "year": 2010 + t, "sales": y[t], "price": rng.normal(),
                         "crisis": int(t >= T - 3)})
    return pd.DataFrame(rows)


def test_paneldata_lag_construction():
    p = fp.PanelData(_toy(), "firm", "year", "sales", y_lags=[1, 2], x={"price": [0, 1]}, trend=True)
    assert p.labels == ["const", "sales(-1)", "sales(-2)", "price", "price(-1)", "trend"]
    Y, W = p.design()
    np.testing.assert_allclose(W[3, 5, 1], Y[3, 4])
    np.testing.assert_allclose(W[3, 5, 2], Y[3, 3])
    assert np.isnan(W[3, 1, 2])


def test_wide_to_long():
    wide = pd.DataFrame({"year": [1, 2, 3], "A": [1., 2., 3.], "B": [4., 5., 6.], "p": [7., 8., 9.]})
    long = fp.wide_to_long(wide, "year", ["A", "B"], "y", "unit", common=["p"])
    assert long.shape == (6, 4) and set(long["unit"]) == {"A", "B"}


def test_forecast_shapes_and_missing_future_error():
    res = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1], x={"price": 1}).fit()
    assert res.forecast(1).shape == (8, len(fp.DEFAULT_METHODS))
    with pytest.raises(ValueError, match="future"):
        res.forecast(3)
    assert res.forecast(3, fill="last").shape == (24, len(fp.DEFAULT_METHODS))


def test_future_period_labels():
    res = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1]).fit()
    assert list(res.forecast(3).loc["F0"].index) == [2024, 2025, 2026]


def test_scenario_variable_changes_forecast():
    m = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1], exog=["crisis"], methods=["pooled"])
    res = m.fit()
    f0 = res.forecast(2, future=pd.DataFrame({"year": [2024, 2025], "crisis": [0, 0]}))
    f1 = res.forecast(2, future=pd.DataFrame({"year": [2024, 2025], "crisis": [1, 1]}))
    assert not np.allclose(f0.to_numpy(), f1.to_numpy())


def test_one_step_forecast_equals_engine():
    p = fp.PanelData(_toy(), "firm", "year", "sales", y_lags=[1], x={"price": 1})
    res = fp.PanelForecaster.from_panel(p).fit()
    Y, W, _ = p.window(p.Tall - 1)
    wn = np.c_[np.ones(p.N), Y[:, -1], p._x["price"][:, -1]]
    fit = fp.panel_forecasts(Y, W, wn)
    np.testing.assert_allclose(res.forecast(1)["Emp. Bayes"].to_numpy(), fit.forecast["eb"], atol=1e-10)


def test_offsets():
    res = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1], methods=["pooled"]).fit()
    a = res.forecast(2)
    b = res.forecast(2, offsets={"F0": [1.0, 2.0]})
    np.testing.assert_allclose((b - a).loc["F0"]["Pooled"].to_numpy(), [1.0, 2.0])
    np.testing.assert_allclose((b - a).loc["F1"]["Pooled"].to_numpy(), [0.0, 0.0])


def test_intervals_reproducible_and_ordered():
    res = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1], methods=["pooled", "eb"]).fit()
    a = res.intervals(2, n_boot=30, seed=5)
    b = res.intervals(2, n_boot=30, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert (a[("Emp. Bayes", "lower")] <= a[("Emp. Bayes", "upper")]).all()


def test_evaluation_expanding_small_T():
    m = fp.PanelForecaster(_toy(N=9, T=12), "firm", "year", "sales", y_lags=[1])
    ev = m.evaluate(window=6, scheme="expanding")
    assert ev.msfe_table().loc["Individual"].iloc[0] > 0
    assert "Panel DM" in ev.dm_tests().index
    assert "Out-of-sample" in str(ev.summary())


def test_evaluation_multistep():
    m = fp.PanelForecaster(_toy(N=9, T=14), "firm", "year", "sales", y_lags=[1], methods=["pooled"])
    ev = m.evaluate(window=7, horizon=2)
    assert ev.horizon == 2 and np.isfinite(ev.errors).any()


def test_summary_and_latex():
    m = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1])
    s = str(m.fit().summary())
    assert "Comb. (pool)" in s and "Mean group" in s
    ev = m.evaluate(window=6)
    assert "\\toprule" in ev.to_latex() and "\\toprule" in ev.to_latex("dm")


def test_plots_return_fig_ax():
    import matplotlib
    matplotlib.use("Agg")
    m = fp.PanelForecaster(_toy(), "firm", "year", "sales", y_lags=[1])
    res = m.fit()
    res.forecast(2, fill="last")
    for f in (res.plot_forecasts, res.plot_coefficients, res.plot_weights):
        fig, _ = f()
        assert fig is not None
    ev = m.evaluate(window=6)
    for f in (ev.plot_density, ev.plot_heatmap, ev.plot_cumulative):
        fig, _ = f()
        assert fig is not None


def test_bad_inputs():
    with pytest.raises(KeyError):
        fp.PanelData(_toy(), "firm", "year", "nope")
    with pytest.raises(ValueError):
        fp.panel_forecasts(np.zeros((3, 2)), np.zeros((3, 2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="Unknown method"):
        fp.panel_forecasts(np.zeros((3, 9)), np.ones((3, 9, 2)), np.ones((3, 2)), ["magic"])
