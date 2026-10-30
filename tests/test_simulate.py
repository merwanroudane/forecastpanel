import numpy as np
import pytest

import forecastpanel as fp


@pytest.mark.parametrize("r0,ab,s2g,rc,expected", [
    (0.775, 0.0, 0.0, 0.0, 0.605), (0.688, 0.5, 0.1, 0.0, 0.640), (0.486, 1.0, 0.2, 0.0, 0.669),
    (0.775, 0.0, 0.0, 0.5, 0.605), (0.688, 0.5, 0.1, 0.5, 0.651), (0.486, 1.0, 0.2, 0.5, 0.686)])
def test_pr2_table_S1(r0, ab, s2g, rc, expected):
    assert abs(fp.panel_pr2(r0, ab, (0.1, 2 / 3, 4 / 3), s2g, rc) - expected) < 6e-4


def test_monte_carlo_reproducible():
    a = fp.monte_carlo(N=40, T=20, n_rep=4, seed=3)
    b = fp.monte_carlo(N=40, T=20, n_rep=4, seed=3)
    assert a.equals(b) and a["Individual"] == 1.0


@pytest.mark.slow
def test_monte_carlo_close_to_table1():
    """Table 1, first row (a_beta=0, s2_alpha=0.5, kappa=0, rho=0, T=20); 300 reps, tol 0.03."""
    r = fp.monte_carlo(N=100, T=20, n_rep=300, seed=11)
    paper = {"Pooled": 0.864, "RE": 0.911, "FE": 0.923, "Emp. Bayes": 0.935,
             "Comb. (pool)": 0.913, "Comb. (FE)": 0.955, "Comb. (w*_i)": 0.947}
    for k, v in paper.items():
        assert abs(r[k] - v) < 0.03, (k, r[k], v)


def test_oracle_weights_in_unit_interval():
    par = fp.draw_parameters(60, a_beta=0.5, sigma2_alpha=0.5, sigma2_gamma=0.1, rho0=0.688,
                             rng=np.random.default_rng(0))
    Y, W, wn, yn, eps = fp.simulate_panel(par, 30, np.random.default_rng(1))
    w = fp.oracle_weights(W, wn, par.theta, eps)
    assert 0 <= w["pool"] <= 1 and 0 <= w["fe"] <= 1


def test_pca_recovers_factor():
    rng = np.random.default_rng(0)
    f = rng.normal(size=80)
    X = np.outer(f, rng.normal(size=30)) + 0.2 * rng.normal(size=(80, 30))
    X[rng.random(X.shape) < 0.05] = np.nan
    F = fp.principal_components(X, 1)
    assert abs(np.corrcoef(F[:, 0], f)[0, 1]) > 0.98
