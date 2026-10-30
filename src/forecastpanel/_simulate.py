"""Monte Carlo design of Section 5.1 (``randPanelData16.m``, ``MCpanelARX.m``).

y_it = alpha_i + beta_i y_{i,t-1} + gamma_i x_it + eps_it                  (47)
x_it = mu_xi + xi_it, xi_it = rho_xi xi_{i,t-1} + s_xi (1-rho_xi^2)^{1/2} nu_it (48)
alpha_i = alpha_0i + phi mu_xi + s_eta eta_i,
gamma_i = gamma_0i + pi mu_xi + s_zeta zeta_i                              (49)

As in the code, the unit parameters are drawn ONCE and kept fixed across
replications; only the data are redrawn. Python's generator cannot reproduce
MATLAB's legacy ``randn('seed')`` streams, so results agree with Table 1 up to
Monte Carlo error, not bit for bit.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._engine import _resolve, method_label, panel_forecasts
from ._core import oracle_weights

__all__ = ["DGPParameters", "draw_parameters", "simulate_panel", "monte_carlo",
           "table1_designs", "panel_pr2", "delta_ar_numerical"]

#: (a_beta, sigma2_alpha, sigma2_gamma, rho0) of the three heterogeneity settings;
#: rho0 calibrated to a pooled R^2 of about 0.6 (Table S.1).
TABLE1_SETTINGS = [(0.0, 0.5, 0.0, 0.775), (0.5, 0.5, 0.1, 0.688), (1.0, 1.0, 0.2, 0.486)]


@dataclass
class DGPParameters:
    alpha: np.ndarray
    mu_x: np.ndarray
    beta: np.ndarray       # AR coefficient rho_y,i
    rho_x: np.ndarray
    gamma: np.ndarray
    sigma2_x: np.ndarray
    sigma2_e: np.ndarray
    w_next: np.ndarray     # (N, 2): (y_iT, x_i,T+1) in the forecast period
    kappa: np.ndarray

    @property
    def theta(self) -> np.ndarray:
        return np.column_stack([self.alpha, self.beta, self.gamma])


def draw_parameters(N: int, a_beta: float = 0.0, sigma2_alpha: float = 0.5,
                    sigma2_gamma: float = 0.0, rho0: float = 0.775, rho_corr: float = 0.0,
                    gamma0: float = 0.1, kappa="zero", x_shift: str = "variance",
                    rng: Optional[np.random.Generator] = None) -> DGPParameters:
    """Step 1 of ``randPanelData16.m``: unit-specific parameters and w_{i,T+1}.

    Parameters
    ----------
    rho_corr : float
        Correlation of alpha_i and gamma_i with the regressor (code: Rbx = 0.5
        when correlated); phi = rho sigma_alpha, pi = rho sigma_gamma (50)-(51).
    kappa : {'zero', 'pm1'} or array
        kappa_i = 0, or +1 for the first half of the units and -1 for the rest.
    x_shift : {'variance', 'sd'}
        The code sets x_{i,T+1} = mu_xi + kappa_i sigma_xi^2 (variance); the text
        says sigma_xi. Default follows the code.  # PAPER vs CODE
    """
    rng = np.random.default_rng() if rng is None else rng
    if isinstance(kappa, str):
        kap = np.zeros(N) if kappa == "zero" else np.r_[np.ones(N // 2), -np.ones(N - N // 2)]
    else:
        kap = np.asarray(kappa, float)
    half = np.arange(1, N + 1) <= N / 2
    g0i = np.where(half, 0.2 / 3, 0.4 / 3) if sigma2_gamma > 0 else np.full(N, gamma0)
    a0i = np.where(half, 2 / 3, 4 / 3) if sigma2_alpha > 0 else np.ones(N)
    phi = rho_corr * np.sqrt(sigma2_alpha)
    ppi = rho_corr * np.sqrt(sigma2_gamma)
    mux = (rng.standard_normal(N) ** 2 - 1) / np.sqrt(2)
    alpha = a0i + phi * mux + np.sqrt(sigma2_alpha - phi ** 2) * rng.standard_normal(N)
    gamma = g0i + ppi * mux + np.sqrt(max(sigma2_gamma - ppi ** 2, 0)) * rng.standard_normal(N)
    rho_x = rng.uniform(size=N) * 0.95
    beta = np.clip(rho0 + rng.uniform(size=N) * a_beta - a_beta / 2, -0.986, 0.986)
    s2e = 0.5 + 0.5 * rng.standard_normal(N) ** 2
    s2x = 0.5 + 0.5 * rng.standard_normal(N) ** 2
    vterm = gamma ** 2 * s2x / (1 - beta ** 2) * (1 + 2 * beta * rho_x / (1 - beta * rho_x))
    wy = (alpha + mux * gamma) / (1 - beta) + kap * np.sqrt(s2e / (1 - beta ** 2) + vterm)
    wx = mux + kap * (s2x if x_shift == "variance" else np.sqrt(s2x))
    return DGPParameters(alpha, mux, beta, rho_x, gamma, s2x, s2e, np.column_stack([wy, wx]), kap)


def simulate_panel(par: DGPParameters, T: int, rng: Optional[np.random.Generator] = None):
    """Step 2: one data set. Returns ``Y (N,T), W (N,T,3), w_next (N,3), y_next (N,), eps (N,T)``.

    y_{i0} is drawn from the stationary distribution, xi_{i0} = 0, and the
    target y_{i,T+1} is generated from w_{i,T+1} as in the code.
    """
    rng = np.random.default_rng() if rng is None else rng
    N = par.alpha.size
    n = T + 2                                     # initial period + T + target
    nu = rng.standard_normal((n, N))
    eps = (rng.standard_normal((n, N)) ** 2 - 1) / np.sqrt(2)
    y = np.empty((n, N))
    x = np.full((n, N), np.nan)
    xi = np.zeros(N)
    s2i0 = (par.gamma ** 2 * par.sigma2_x + par.sigma2_e) / (1 - par.beta ** 2)
    y[0] = (par.alpha + par.gamma * par.mu_x) / (1 - par.beta) + np.sqrt(s2i0) * rng.standard_normal(N)
    se = np.sqrt(par.sigma2_e)
    for t in range(1, n):
        xi = par.rho_x * xi + np.sqrt(par.sigma2_x) * np.sqrt(1 - par.rho_x ** 2) * nu[t]
        x[t] = par.mu_x + xi
        y[t] = par.alpha + par.beta * y[t - 1] + par.gamma * x[t] + se * eps[t]
    y_next = par.alpha + par.beta * par.w_next[:, 0] + par.gamma * par.w_next[:, 1] + se * eps[-1]
    Y = y[1:T + 1].T
    W = np.stack([np.ones((N, T)), y[0:T].T, x[1:T + 1].T], axis=2)
    w_next = np.c_[np.ones(N), par.w_next]
    return Y, W, w_next, y_next, (se[:, None] * eps[1:T + 1].T)


def monte_carlo(N: int = 100, T: int = 20, n_rep: int = 1000, a_beta: float = 0.0,
                sigma2_alpha: float = 0.5, sigma2_gamma: float = 0.0, rho0: float = 0.775,
                rho_corr: float = 0.0, kappa: str = "zero", methods=None, oracle: bool = False,
                seed: int = 10000, param_seed: Optional[int] = None, n_jobs: int = 1,
                **engine_kw) -> pd.Series:
    """Relative MSFE of each method over ``n_rep`` replications (one cell of Table 1).

    rMSFE_j = sum_i sum_r (y - yhat_j)^2 / sum_i sum_r (y - yhat_ind)^2.

    Examples
    --------
    >>> from forecastpanel import monte_carlo
    >>> r = monte_carlo(N=50, T=20, n_rep=5, seed=1)
    >>> round(float(r["Individual"]), 3)
    1.0
    """
    meths = _resolve(methods)
    prng = np.random.default_rng(seed if param_seed is None else param_seed)
    par = draw_parameters(N, a_beta, sigma2_alpha, sigma2_gamma, rho0, rho_corr, kappa=kappa, rng=prng)
    seeds = np.random.SeedSequence(seed).spawn(n_rep)

    def one(ss):
        rng = np.random.default_rng(ss)
        Y, W, wn, yn, eps = simulate_panel(par, T, rng)
        fit = panel_forecasts(Y, W, wn, meths, rng=rng, **engine_kw)
        se = {m: (yn - fit.forecast[m]) ** 2 for m in meths}
        if oracle:
            ow = oracle_weights(W, wn, par.theta, eps)
            fp_, ff_ = fit.forecast, None
            pool = (wn * np.tile(np.linalg.pinv(np.einsum("itk,itl->kl", W, W)) @
                                 np.einsum("itk,it->k", W, Y), (N, 1))).sum(axis=1)
            from ._core import fixed_effects
            fe = (wn * fixed_effects(Y, W).theta).sum(axis=1)
            ind = fp_["individual"]
            se["oracle_pool"] = (yn - (ow["pool"] * ind + (1 - ow["pool"]) * pool)) ** 2
            se["oracle_fe"] = (yn - (ow["fe"] * ind + (1 - ow["fe"]) * fe)) ** 2
        return {k: v.sum() for k, v in se.items()}

    if n_jobs == 1:
        out = [one(s) for s in seeds]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in seeds)
    tot = pd.DataFrame(out).sum()
    rel = tot / tot["individual"]
    rel.index = [method_label(m) if m in meths else m for m in rel.index]
    rel.name = f"N={N},T={T}"
    return rel


def table1_designs(N: int = 100, T: Sequence[int] = (20, 50, 100)) -> pd.DataFrame:
    """The 36 parameter configurations of Table 1 (per kappa and N)."""
    rows = []
    for kap in ("zero", "pm1"):
        for rc in (0.0, 0.5):
            for (ab, s2a, s2g, r0) in TABLE1_SETTINGS:
                for t in T:
                    rows.append(dict(kappa=kap, rho_corr=rc, a_beta=ab, sigma2_alpha=s2a,
                                     sigma2_gamma=s2g, rho0=r0, N=N, T=t))
    return pd.DataFrame(rows)


def panel_pr2(rho0: float, a_rho: float, gamma0=(0.1, 2 / 3, 4 / 3), sigma2_gamma: float = 0.0,
              rho_corr: float = 0.0) -> float:
    """Pooled R^2 of the Monte Carlo design (Supplement S.2, ``panelPR2.m``)."""
    Vg = sigma2_gamma * (rho_corr ** 2 / (1 - rho_corr ** 2) + 1)
    if np.isscalar(gamma0):
        Eg2 = gamma0 ** 2 + Vg
    else:
        g = list(gamma0)
        Eg2 = g[0] ** 2 * (g[1] ** 2 / 2 + g[2] ** 2 / 2) + Vg
    if a_rho == 0:
        Er2 = 1 / (1 - rho0 ** 2)
    else:
        Er2 = 1 / (2 * a_rho) * (np.log((1 + rho0 + a_rho / 2) / (1 + rho0 - a_rho / 2))
                                 - np.log((1 - rho0 - a_rho / 2) / (1 - rho0 + a_rho / 2)))
    return float((Eg2 * Er2 + Er2 - 1) / (Eg2 * Er2 + Er2))


def delta_ar_numerical(beta0: float, n_draws: int = 10000, seed: int = 12345):
    """Delta_AR and E(y^2 eta) of the panel AR(1) example (Appendix D, Table A.1).

    eta_i ~ U(-1/2, 1/2), sigma^2 = 1 (``DeltaARnumerical.m``).
    """
    eta = np.random.default_rng(seed).uniform(size=n_draws) - 0.5
    p1 = np.mean(eta ** 2 / (1 - (beta0 + eta) ** 2))
    p3 = np.mean(eta / (1 - (beta0 + eta) ** 2))
    p2 = 0.5 * (np.log((1 + beta0 + 0.5) / (1 + beta0 - 0.5)) - np.log((1 - beta0 - 0.5) / (1 - beta0 + 0.5)))
    delta = (p1 * p2 - p3 ** 2) / p2
    ey2 = 0.5 * (-(1 - beta0) * np.log((1 - beta0 - 0.5) / (1 - beta0 + 0.5))
                 - (1 + beta0) * np.log((1 + beta0 + 0.5) / (1 + beta0 - 0.5)))
    return float(ey2), float(delta)
