"""One-window forecasting engine: all methods on a balanced estimation window.

``panel_forecasts`` takes arrays in the paper's notation and returns, for every
method, the unit-specific coefficient vector theta_i^(m) that generates the
forecast y_hat_{i,T+1} = w_{i,T+1}' theta_i^(m). Writing every method in this
form (combinations combine coefficients with the same weights as the
forecasts) gives identical one-step forecasts to the replication code and lets
multi-step forecasts be iterated for every method.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import _core as core

# key -> (label used in tables, short description)
METHODS: Dict[str, tuple] = {
    "individual": ("Individual", "Unit-by-unit OLS, eq. (11) - benchmark"),
    "pooled": ("Pooled", "Pooled OLS, eq. (12)"),
    "re": ("RE", "Random-effects GLS + Goldberger BLUP, eq. (S.16)"),
    "fe": ("FE", "Fixed effects, eq. (25)"),
    "eb": ("Emp. Bayes", "Empirical Bayes (Hsiao et al. 1999), eq. (44)"),
    "comb_pool": ("Comb. (pool)", "Individual/pooled, weight eq. (32)"),
    "comb_fe": ("Comb. (FE)", "Individual/FE, weight eq. (39)"),
    "comb_unit": ("Comb. (w*_i)", "Unit-specific weights, eq. (36)/(46)"),
    "eq_pool": ("Eq.w. (pool)", "Equal weights individual/pooled (Table S.4)"),
    "eq_fe": ("Eq.w. (FE)", "Equal weights individual/FE (Table S.4)"),
    # ---- extensions beyond the paper ----
    "comb_unit_ba": ("Comb. (w*_i, BA)", "Bias-adjusted unit weights (PPT 2022)"),
    "mean_group": ("Mean group", "Mean-group estimator (Pesaran-Smith 1995)"),
    "ib": ("Iter. Bayes", "Iterative empirical Bayes (Maddala et al. 1997)"),
    "stein": ("Stein shrink.", "Stein-rule shrinkage (Maddala et al. 1997)"),
    "mallows": ("Mallows avg.", "Mallows pooling averaging (Wang et al. 2019)"),
    "re_ar1": ("RE-AR(1)", "RE BLUP with AR(1) errors (Baltagi-Li 1992)"),
    "hb": ("Hier. Bayes", "Hierarchical Bayes, Gibbs (Lindley-Smith 1972)"),
}

#: Methods of Table 1-3 of the paper, in the paper's order.
PAPER_METHODS: List[str] = ["individual", "pooled", "re", "fe", "eb",
                            "comb_pool", "comb_fe", "comb_unit"]
#: Paper methods plus the equal-weight combinations of the Supplement.
DEFAULT_METHODS: List[str] = PAPER_METHODS + ["eq_pool", "eq_fe"]
ALL_METHODS: List[str] = list(METHODS)


def method_label(key: str) -> str:
    return METHODS[key][0] if key in METHODS else key


@dataclass
class PanelFit:
    """Result of fitting every requested method on one estimation window."""

    methods: List[str]
    theta: Dict[str, np.ndarray]           # method -> (N, K)
    forecast: Dict[str, np.ndarray]        # method -> (N,) one-step forecast
    weights: Dict[str, object] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)
    N: int = 0
    T: int = 0
    K: int = 0

    def forecast_matrix(self) -> np.ndarray:
        return np.column_stack([self.forecast[m] for m in self.methods])


def _resolve(methods: Optional[Sequence[str]]) -> List[str]:
    if methods is None:
        return list(DEFAULT_METHODS)
    if isinstance(methods, str):
        if methods == "paper":
            return list(PAPER_METHODS)
        if methods == "all":
            return [m for m in ALL_METHODS if m != "hb"]
        methods = [methods]
    out = []
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from {sorted(METHODS)}.")
        out.append(m)
    if "individual" not in out:
        out.insert(0, "individual")
    return out


def panel_forecasts(Y: np.ndarray, W: np.ndarray, w_next: np.ndarray,
                    methods: Optional[Sequence[str]] = None, *,
                    fe_weight: str = "code", unit_sigma: str = "ppt2022",
                    omega_regularize: Optional[str] = None,
                    hb_options: Optional[dict] = None,
                    rng: Optional[np.random.Generator] = None) -> PanelFit:
    """Fit all requested panel forecasting methods on one balanced window.

    Parameters
    ----------
    Y : ndarray (N, T)
        Outcomes of the estimation window.
    W : ndarray (N, T, K)
        Regressors including the intercept in column 0.
    w_next : ndarray (N, K)
        Regressors for the forecast period, w_{i,T+1}.
    methods : list of str, 'paper', 'all' or None
        Method keys (see ``forecastpanel.METHODS``). ``None`` gives the paper's
        eight methods plus the two equal-weight combinations.
    fe_weight : {'code', 'paper'}
        Estimator of the individual/FE weight (see ``weight_individual_fe``).
    unit_sigma : {'ppt2022', 'ols'}
        Error-variance estimator in the unit-specific weights.
    omega_regularize : {None, 'ridge', 'diagonal'}
        Regularisation of Omega_eta in empirical Bayes (small N extension).
    hb_options : dict
        Keyword arguments of ``hierarchical_bayes``.

    Returns
    -------
    PanelFit

    Examples
    --------
    >>> import numpy as np
    >>> from forecastpanel import panel_forecasts
    >>> rng = np.random.default_rng(0)
    >>> N, T = 30, 20
    >>> x = rng.normal(size=(N, T + 1))
    >>> y = 1 + 0.5 * x + rng.normal(size=(N, T + 1))
    >>> W = np.stack([np.ones((N, T)), x[:, :T]], axis=2)
    >>> fit = panel_forecasts(y[:, :T], W, np.c_[np.ones(N), x[:, T]])
    >>> sorted(fit.forecast)[:3]
    ['comb_fe', 'comb_pool', 'comb_unit']
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    w_next = np.asarray(w_next, dtype=float)
    N, T, K = W.shape
    if Y.shape != (N, T) or w_next.shape != (N, K):
        raise ValueError("Shapes must be Y (N,T), W (N,T,K), w_next (N,K).")
    if T <= K:
        raise ValueError(f"T={T} observations per unit but K={K} coefficients: "
                         "individual regressions are not identified.")
    meths = _resolve(methods)
    ind = core.individual_ols(Y, W)
    theta: Dict[str, np.ndarray] = {"individual": ind.theta}
    weights: Dict[str, object] = {}
    details: Dict[str, object] = {"individual": ind}
    need = set(meths)

    tp = None
    if need & {"pooled", "comb_pool", "comb_unit", "comb_unit_ba", "eq_pool", "stein", "mallows"}:
        tp = core.pooled_ols(Y, W)
        theta["pooled"] = np.tile(tp, (N, 1))
    fe = None
    if need & {"fe", "comb_fe", "eq_fe"}:
        fe = core.fixed_effects(Y, W)
        theta["fe"] = fe.theta
        details["fe"] = fe
    if "re" in need:
        re = core.random_effects_blup(Y, W)
        theta["re"] = re.theta
        details["re"] = re
    if "re_ar1" in need:
        ra = core.random_effects_ar1_blup(Y, W)
        th = np.tile(ra.theta_gls, (N, 1))
        th[:, 0] += ra.correction(1)
        theta["re_ar1"] = th
        details["re_ar1"] = ra
    if "eb" in need:
        eb = core.empirical_bayes(Y, W, ind, omega_regularize)
        theta["eb"] = eb.theta
        details["eb"] = eb
    if "ib" in need:
        ib = core.iterative_bayes(Y, W, omega_regularize=omega_regularize)
        theta["ib"] = ib.theta
        details["ib"] = ib
    if "mean_group" in need:
        theta["mean_group"] = np.tile(core.mean_group(ind.theta), (N, 1))
    if "comb_pool" in need:
        wr = core.weight_individual_pooled(Y, W, w_next, ind, tp)
        theta["comb_pool"] = wr.weight * ind.theta + (1 - wr.weight) * theta["pooled"]
        weights["comb_pool"] = wr
    if "comb_fe" in need:
        wr = core.weight_individual_fe(Y, W, w_next, ind, fe, variant=fe_weight)
        theta["comb_fe"] = wr.weight * ind.theta + (1 - wr.weight) * fe.theta
        weights["comb_fe"] = wr
    if "comb_unit" in need:
        uw = core.weights_unit_specific(Y, W, w_next, ind, sigma=unit_sigma)
        theta["comb_unit"] = uw.weights[:, None] * ind.theta + (1 - uw.weights[:, None]) * theta["pooled"]
        weights["comb_unit"] = uw
    if "comb_unit_ba" in need:
        uw = core.weights_unit_specific(Y, W, w_next, ind, sigma=unit_sigma, bias_adjusted=True)
        theta["comb_unit_ba"] = uw.weights[:, None] * ind.theta + (1 - uw.weights[:, None]) * theta["pooled"]
        weights["comb_unit_ba"] = uw
    if "eq_pool" in need:
        theta["eq_pool"] = 0.5 * ind.theta + 0.5 * theta["pooled"]
    if "eq_fe" in need:
        theta["eq_fe"] = 0.5 * ind.theta + 0.5 * fe.theta
    if "stein" in need:
        wr = core.weight_stein(Y, W, ind, tp)
        theta["stein"] = (1 - wr.weight) * ind.theta + wr.weight * theta["pooled"]
        weights["stein"] = wr
    if "mallows" in need:
        wr = core.weight_mallows(Y, W, ind)
        theta["mallows"] = (1 - wr.weight) * ind.theta + wr.weight * theta["pooled"]
        weights["mallows"] = wr
    if "hb" in need:
        yhb, thb = core.hierarchical_bayes(Y, W, w_next, rng=rng, **(hb_options or {}))
        theta["hb"] = thb
        details["hb_forecast"] = yhb

    forecast = {m: (w_next * theta[m]).sum(axis=1) for m in meths}
    if "hb" in meths:
        forecast["hb"] = details["hb_forecast"]
    return PanelFit(meths, {m: theta[m] for m in meths}, forecast, weights, details, N, T, K)
