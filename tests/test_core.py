import numpy as np
import pytest

import forecastpanel as fp
from forecastpanel import _core as core


def test_individual_equals_lstsq(small_panel):
    Y, W, wn, _ = small_panel
    ind = fp.individual_ols(Y, W)
    for i in range(Y.shape[0]):
        np.testing.assert_allclose(ind.theta[i], np.linalg.lstsq(W[i], Y[i], rcond=None)[0], atol=1e-10)


def test_pooled_equals_stacked_ols(small_panel):
    Y, W, wn, _ = small_panel
    N, T, K = W.shape
    b = np.linalg.lstsq(W.reshape(N * T, K), Y.reshape(-1), rcond=None)[0]
    np.testing.assert_allclose(fp.pooled_ols(Y, W), b, atol=1e-10)


def test_fe_equals_dummy_regression(small_panel):
    Y, W, wn, _ = small_panel
    N, T, K = W.shape
    D = np.kron(np.eye(N), np.ones((T, 1)))
    X = np.c_[D, W[:, :, 1:].reshape(N * T, K - 1)]
    b = np.linalg.lstsq(X, Y.reshape(-1), rcond=None)[0]
    fe = fp.fixed_effects(Y, W)
    np.testing.assert_allclose(fe.beta, b[N:], atol=1e-9)
    np.testing.assert_allclose(fe.alpha, b[:N], atol=1e-9)


def test_eb_is_matrix_weighted_average(small_panel):
    """eq. (45): theta_EB = W_iT theta_i + (I - W_iT) theta_bar."""
    Y, W, wn, _ = small_panel
    ind = fp.individual_ols(Y, W)
    eb = fp.empirical_bayes(Y, W, ind)
    K = W.shape[2]
    for i in range(Y.shape[0]):
        alt = eb.shrinkage[i] @ ind.theta[i] + (np.eye(K) - eb.shrinkage[i]) @ eb.theta_bar
        np.testing.assert_allclose(eb.theta[i], alt, atol=1e-9)


def test_eb_tends_to_individual_when_noise_vanishes(small_panel):
    Y, W, wn, _ = small_panel
    ind = fp.individual_ols(Y, W)
    tiny = core.IndividualFit(ind.theta, ind.sigma2 * 1e-10, ind.resid, ind.WW, ind.Wy, ind.rank_deficient)
    eb = fp.empirical_bayes(Y, W, tiny)
    np.testing.assert_allclose(eb.theta, ind.theta, atol=1e-6)


def test_halves_drop_middle_observation():
    a, b = core._halves(7)
    assert list(a) == [0, 1, 2] and list(b) == [4, 5, 6]
    a, b = core._halves(8)
    assert list(a) == [0, 1, 2, 3] and list(b) == [4, 5, 6, 7]


def test_pooled_weight_components_formula(small_panel):
    """omega = (Delta - psi/T) / (Delta + h/T - 2 psi/T), eq. (32)."""
    Y, W, wn, _ = small_panel
    ind = fp.individual_ols(Y, W)
    wr = fp.weight_individual_pooled(Y, W, wn, ind, fp.pooled_ols(Y, W))
    c = wr.components
    T = c["T"]
    assert np.isclose(wr.raw, (c["Delta"] - c["psi"] / T) / (c["Delta"] + c["h"] / T - 2 * c["psi"] / T))
    assert 0 <= wr.weight <= 1


def test_homogeneous_panel_puts_weight_on_pooling():
    rng = np.random.default_rng(3)
    N, T = 200, 15
    x = rng.normal(size=(N, T + 1))
    y = 1 + 0.5 * x + rng.normal(size=(N, T + 1))
    W = np.stack([np.ones((N, T)), x[:, :T]], axis=2)
    fit = fp.panel_forecasts(y[:, :T], W, np.c_[np.ones(N), x[:, T]], ["comb_pool", "mallows", "stein"])
    # Delta_hat (40) contains an O(1/T) estimation-noise term (Appendix C), so with
    # homogeneous slopes and small T the estimated weight stays near 1/2, not 0.
    assert fit.weights["comb_pool"].weight < 0.6
    assert fit.weights["mallows"].weight > 0.7        # weight on the pooled estimator
    assert fit.weights["stein"].weight > 0.7


def test_fe_weight_variants(small_panel):
    Y, W, wn, _ = small_panel
    ind = fp.individual_ols(Y, W)
    fe = fp.fixed_effects(Y, W)
    a = fp.weight_individual_fe(Y, W, wn, ind, fe, variant="code")
    b = fp.weight_individual_fe(Y, W, wn, ind, fe, variant="paper")
    assert 0 <= a.weight <= 1 and 0 <= b.weight <= 1
    with pytest.raises(ValueError):
        fp.weight_individual_fe(Y, W, wn, ind, fe, variant="x")


def test_re_ar1_recovers_rho():
    rng = np.random.default_rng(7)
    N, T, rho = 300, 30, 0.6
    mu = rng.normal(size=N)
    v = np.zeros((N, T))
    e = rng.normal(size=(N, T))
    v[:, 0] = e[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        v[:, t] = rho * v[:, t - 1] + e[:, t]
    x = rng.normal(size=(N, T))
    y = 1 + 0.5 * x + mu[:, None] + v
    W = np.stack([np.ones((N, T)), x], axis=2)
    fit = fp.random_effects_ar1_blup(y, W)
    assert abs(fit.rho - rho) < 0.08          # within-residual estimator has O(1/T) bias
    assert abs(fit.theta_gls[1] - 0.5) < 0.02


def test_singular_omega_message():
    rng = np.random.default_rng(0)
    N, T = 3, 20
    W = np.concatenate([np.ones((N, T, 1)), rng.normal(size=(N, T, 3))], axis=2)
    Y = rng.normal(size=(N, T))
    with pytest.raises(np.linalg.LinAlgError, match="omega_regularize"):
        fp.empirical_bayes(Y, W)
    eb = fp.empirical_bayes(Y, W, omega_regularize="ridge")
    assert np.all(np.isfinite(eb.theta))


def test_hierarchical_bayes_reproducible(small_panel):
    Y, W, wn, _ = small_panel
    a, _ = fp.hierarchical_bayes(Y, W, wn, n_draws=60, burn_in=20, rng=np.random.default_rng(1))
    b, _ = fp.hierarchical_bayes(Y, W, wn, n_draws=60, burn_in=20, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    ind = (wn * fp.individual_ols(Y, W).theta).sum(axis=1)
    assert np.corrcoef(a, ind)[0, 1] > 0.8
