"""Compatibility proof: agreement with the authors' MATLAB replication code.

* Forecasts of 11 methods on three simulated panels (one with odd T) produced by
  panelRandomCorrCoeffFreqForecasts.m, panelRandomCoeffForecastsOldWeights.m and
  panelEmpiricalBayesForecasts.m (MATLAB R2024a). Tolerance: 1e-10 absolute.
* The house-price application reproduces the published Table 2 (upper panel)
  and Table 3 (upper panel) to the printed decimals.
"""
import numpy as np
import pytest
from conftest import load_xcheck

import forecastpanel as fp

MATLAB_COLUMNS = ["pooled", "re", "individual", "fe", "comb_pool", "comb_fe",
                  "eq_pool", "eq_fe", "comb_unit", "comb_unit_ba", "eb"]


@pytest.mark.parametrize("tag", ["a", "b", "c"])
def test_matches_matlab_forecasts_tol_1e10(tag):
    Y, W, wn, ref = load_xcheck(tag)
    fit = fp.panel_forecasts(Y, W, wn, methods=MATLAB_COLUMNS)
    for j, m in enumerate(MATLAB_COLUMNS):
        np.testing.assert_allclose(fit.forecast[m], ref[:, j], rtol=0, atol=1e-10, err_msg=m)


# Pesaran, Pick and Timmermann (2026), Table 2, upper panel (house prices)
TABLE2_HOUSE = {
    "Individual": [2.822, 2.520, 3.542, np.nan, np.nan, np.nan, 0.008, 0.273, 0.146, 0.569, 0.282, 0.420],
    "Pooled": [0.920, 1.162, 0.947, 0.613, 0.381, 0.536, 0.116, 0.171, 0.249, 0.157, 0.188, 0.160],
    "RE": [0.924, 1.166, 0.960, 0.619, 0.376, 0.528, 0.108, 0.022, 0.047, 0.003, 0.019, 0.008],
    "FE": [0.936, 1.186, 0.980, 0.591, 0.381, 0.517, 0.055, 0.108, 0.105, 0.251, 0.376, 0.296],
    "Emp. Bayes": [0.901, 0.955, 0.881, 0.942, 0.519, 0.652, 0.185, 0.157, 0.127, 0.003, 0.064, 0.052],
    "Comb. (pool)": [0.920, 0.961, 0.932, 0.939, 0.522, 0.688, 0.157, 0.077, 0.099, 0.000, 0.011, 0.017],
    "Comb. (FE)": [0.937, 0.977, 0.940, 0.917, 0.494, 0.677, 0.019, 0.072, 0.069, 0.011, 0.039, 0.044],
    "Comb. (w*_i)": [0.921, 0.957, 0.909, 0.936, 0.541, 0.713, 0.044, 0.028, 0.052, 0.006, 0.003, 0.006],
}
TABLE3_HOUSE = {"Pooled": (-9.45, 60, 6), "RE": (-9.11, 62, 6), "FE": (-7.57, 57, 8),
                "Emp. Bayes": (-24.63, 209, 2), "Comb. (pool)": (-22.93, 189, 0),
                "Comb. (FE)": (-21.19, 169, 0), "Comb. (w*_i)": (-27.59, 240, 0)}


def _house_panel():
    df = fp.load_house_prices(with_neighbors_only=True)
    return fp.PanelData(df, "msa", "quarter", "dhp", y_lags=[1],
                        x={"dhp_spatial": 1, "dhp_region": 1, "dhp_us": 1})


@pytest.fixture(scope="module")
def house_eval():
    return fp.evaluate_forecasts(_house_panel(), window=60, n_forecasts=128, kappa_legacy=True)


@pytest.mark.slow
def test_house_table2_published(house_eval):
    # best/worst shares are taken among the 10 methods of the code (incl. equal weights)
    tab = house_eval.msfe_table(methods=fp.PAPER_METHODS)
    for name, vals in TABLE2_HOUSE.items():
        np.testing.assert_allclose(tab.loc[name].to_numpy(dtype=float), vals, atol=6e-4, err_msg=name)


@pytest.mark.slow
def test_house_table3_published(house_eval):
    dm = house_eval.dm_tests(methods=fp.PAPER_METHODS)
    for name, (stat, lo, hi) in TABLE3_HOUSE.items():
        assert abs(dm.loc["Panel DM", name] - stat) < 6e-3, name
        assert dm.loc["DM<-1.96", name] == lo and dm.loc["DM>1.96", name] == hi, name


@pytest.mark.slow
def test_house_kappa_fix_changes_only_conditional_columns(house_eval):
    ev = fp.evaluate_forecasts(_house_panel(), window=60, n_forecasts=128)
    a = ev.msfe_table(methods=fp.PAPER_METHODS)
    b = house_eval.msfe_table(methods=fp.PAPER_METHODS)
    np.testing.assert_allclose(a.iloc[:, [0, 3]].to_numpy(float), b.iloc[:, [0, 3]].to_numpy(float))
    assert abs(a.loc["Individual"].iloc[1] - 2.258) < 1e-3     # corrected kappa_i = 0
    assert abs(a.loc["Pooled"].iloc[1] - 0.769) < 1e-3
