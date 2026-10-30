"""Pseudo out-of-sample evaluation (Section 6 of the paper).

Rolling or expanding estimation windows; at every forecast origin all methods
are fitted on the units with complete data in the window and the forecasts of
y_{i,t+1} are stored. The results object builds Table 2 (relative MSFE,
frequency of beating the benchmark, of being best and worst, overall and
conditional on the predictor being near its mean, kappa_i = 0, or one standard
deviation away, kappa_i = +/-1), Table 3 (panel and unit DM tests) and Figure 1
(densities of unit-level MSFE ratios).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._engine import _resolve, method_label, panel_forecasts
from ._dm import dm_test, panel_dm_test
from ._panel import PanelData

__all__ = ["evaluate_forecasts", "EvaluationResults"]

_COND = {"all": "all", "kappa0": "kappa_i = 0", "kappa1": "kappa_i = +/-1"}


def evaluate_forecasts(panel: PanelData, window: Optional[int] = 60, scheme: str = "rolling",
                       n_forecasts: Optional[int] = None, first_origin=None, horizon: int = 1,
                       methods: Optional[Sequence[str]] = None, kappa_c: float = 0.1,
                       min_units: int = 3, verbose: bool = False, seed: Optional[int] = None,
                       kappa_legacy: bool = False, **engine_kw) -> "EvaluationResults":
    """Recursive pseudo out-of-sample forecasting exercise.

    Parameters
    ----------
    panel : PanelData
    window : int or None
        Length of the estimation window (paper: 60). With ``scheme='expanding'``
        it is the length of the first window.
    scheme : {'rolling', 'expanding'}
    n_forecasts : int, optional
        Number of forecast origins counted back from the end of the sample;
        default uses every origin for which a full window is available.
    first_origin : period label, optional
        Alternatively, the period of the first forecast origin (the last
        period of the first estimation window).
    horizon : int, default 1
        Forecast horizon; for h > 1 forecasts are iterated, using realised
        values of contemporaneous predictors and the method's own forecasts
        of lagged y.
    methods : list of str
        Methods to evaluate (``'individual'`` is the benchmark and always included).
    kappa_c : float, default 0.1
        Tolerance c for the kappa_i classification (Sec. 6.1).
    kappa_legacy : bool, default False
        Reproduce the house-price kappa columns of Table 2 as published. In
        ``ppt_house.m`` the forecast-period value d_{i,T+1} is stored as a full
        column, so the tabulation compares every unit with unit 1's
        d_{1,T+1}. ``True`` replicates this; the default classifies each unit
        with its own d_{i,T+1}, as Section 6.1 describes.  # PAPER vs CODE
    engine_kw
        Passed to ``panel_forecasts`` (``fe_weight``, ``unit_sigma``, ...).

    Returns
    -------
    EvaluationResults
    """
    meths = _resolve(methods)
    Yall, Wall = panel.design()
    N, Tall, K = Wall.shape
    rng = np.random.default_rng(seed)
    if scheme not in ("rolling", "expanding"):
        raise ValueError("scheme must be 'rolling' or 'expanding'")
    L = window if window is not None else max(K + 2, (Tall - panel.max_lag) // 2)
    first_possible = panel.max_lag + L - 1
    last_origin = Tall - 1 - horizon
    if first_origin is not None:
        start = panel.times.index(first_origin)
    elif n_forecasts is not None:
        start = last_origin - n_forecasts + 1
    else:
        start = first_possible
    if start < first_possible:
        raise ValueError(f"Not enough periods: the first origin needs {L} observations after "
                         f"{panel.max_lag} pre-sample periods.")
    origins = list(range(start, last_origin + 1))
    if not origins:
        raise ValueError("No forecast origin available; reduce `window` or `horizon`.")
    n, M = len(origins), len(meths)
    fc = np.full((n, N, M), np.nan)
    actual = np.full((n, N), np.nan)
    d_mean = np.full((n, N), np.nan)
    d_sd = np.full((n, N), np.nan)
    d_next = np.full((n, N), np.nan)
    weights = {m: np.full(n, np.nan) for m in ("comb_pool", "comb_fe", "stein", "mallows") if m in meths}
    n_units = np.zeros(n, dtype=int)
    for r, e in enumerate(origins):
        s = e - L + 1 if scheme == "rolling" else first_possible - L + 1
        Y, W = Yall[:, s:e + 1], Wall[:, s:e + 1]
        wn = Wall[:, e + 1]
        yt = Yall[:, e + horizon]
        ok = panel.complete_units(Y, W, wn, yt)
        n_units[r] = ok.sum()
        if ok.sum() < min_units:
            continue
        fit = panel_forecasts(Y[ok], W[ok], wn[ok], meths, rng=rng, **engine_kw)
        if horizon == 1:
            fc[r][np.ix_(ok, range(M))] = fit.forecast_matrix()
        else:
            for j, m in enumerate(meths):
                th = np.full((N, K), np.nan)
                th[ok] = fit.theta[m]
                path, _ = panel.path_regressors(th, horizon, panel._x, y_hist=Yall[:, :e + 1])
                fc[r, ok, j] = path[ok, -1]
        actual[r, ok] = yt[ok]
        ind = fit.details["individual"]
        dfit = Y[ok] - ind.resid                       # theta_i'w_it in the window
        d_mean[r, ok] = dfit.mean(axis=1)
        d_sd[r, ok] = dfit.std(axis=1, ddof=1)
        d_next[r, ok] = fit.forecast["individual"]
        for m in weights:
            weights[m][r] = fit.weights[m].weight
        if kappa_legacy:
            first = np.flatnonzero(ok)[0]
            d_next[r, ok] = d_next[r, first]
        if verbose and (r % 10 == 0 or r == n - 1):
            print(f"origin {r + 1}/{n} ({panel.times[e]}): {ok.sum()} units")
    return EvaluationResults(panel, meths, origins, fc, actual, d_mean, d_sd, d_next,
                             kappa_c, weights, n_units, L, scheme, horizon)


class EvaluationResults:
    """Forecasts, errors and accuracy tables of a pseudo out-of-sample exercise."""

    def __init__(self, panel, methods, origins, forecasts, actual, d_mean, d_sd, d_next,
                 kappa_c, weights, n_units, window, scheme, horizon):
        self.panel = panel
        self.methods: List[str] = list(methods)
        self.origins = origins
        self.forecasts = forecasts                    # (n, N, M)
        self.actual = actual                          # (n, N)
        self.errors = actual[:, :, None] - forecasts  # e = y - y_hat (eq. 14)
        self.kappa_c = kappa_c
        self.weights = weights
        self.n_units = n_units
        self.window, self.scheme, self.horizon = window, scheme, horizon
        c = kappa_c
        with np.errstate(invalid="ignore"):
            self.in_mean = np.abs((d_next - d_mean) / (d_sd * c)) < 1
            self.in_std = (np.abs((d_next - d_mean + d_sd) / (c * d_sd)) < 1) | \
                          (np.abs((d_next - d_mean - d_sd) / (c * d_sd)) < 1)

    # ------------------------------------------------------------- accessors
    @property
    def benchmark(self) -> str:
        return "individual"

    @property
    def origin_labels(self) -> list:
        return [self.panel.times[e] for e in self.origins]

    @property
    def target_labels(self) -> list:
        return [self.panel.times[e + self.horizon] for e in self.origins]

    def _j(self, m: str) -> int:
        return self.methods.index(m)

    def _err(self, cond: str) -> np.ndarray:
        e = self.errors
        if cond == "all":
            return e
        mask = self.in_mean if cond == "kappa0" else self.in_std
        return np.where(mask[:, :, None], e, np.nan)

    def unit_msfe(self, cond: str = "all") -> pd.DataFrame:
        """MSFE of every unit (rows) and method (columns)."""
        e = self._err(cond)
        with np.errstate(invalid="ignore"), _quiet():
            ms = np.nanmean(e ** 2, axis=0)
        return pd.DataFrame(ms, index=self.panel.units, columns=self.methods)

    def relative_msfe(self, cond: str = "all") -> pd.DataFrame:
        """Unit MSFE divided by the benchmark's MSFE (input of Figure 1)."""
        ms = self.unit_msfe(cond)
        return ms.div(ms[self.benchmark], axis=0)

    # ---------------------------------------------------------------- Table 2
    def msfe_table(self, methods: Optional[Sequence[str]] = None,
                   best_worst_among: Optional[Sequence[str]] = None,
                   conditions: Sequence[str] = ("all", "kappa0", "kappa1"),
                   scale: float = 1.0) -> pd.DataFrame:
        """Table 2 of the paper.

        Rows are methods. For each conditioning set the columns give the ratio
        of the average MSFE to the benchmark's (first row: the benchmark's
        average MSFE x ``scale``), the fraction of units for which the method
        beats the benchmark, and the fractions for which it has the smallest
        and the largest MSFE among ``best_worst_among`` (default: all
        evaluated methods). Fractions are over all units, as in the code.
        """
        rows = list(methods) if methods is not None else self.methods
        pool = list(best_worst_among) if best_worst_among is not None else self.methods
        b = self.benchmark
        blocks = {}
        for cond in conditions:
            ms = self.unit_msfe(cond)
            arr = ms[pool].to_numpy()
            with np.errstate(invalid="ignore"), _quiet():
                mn, mx = np.nanmin(arr, axis=1), np.nanmax(arr, axis=1)
            ratio, beat, best, worst = {}, {}, {}, {}
            bavg = np.nanmean(ms[b].to_numpy())
            for m in rows:
                v = ms[m].to_numpy()
                ratio[m] = bavg * scale if m == b else np.nanmean(v) / bavg
                with np.errstate(invalid="ignore"):
                    beat[m] = np.nan if m == b else float(np.mean(v < ms[b].to_numpy()))
                    best[m] = float(np.mean(v == mn)) if m in pool else np.nan
                    worst[m] = float(np.mean(v == mx)) if m in pool else np.nan
            lab = _COND[cond]
            blocks[("Ratio of ave. MSFE", lab)] = ratio
            blocks[("Freq. beating benchmark", lab)] = beat
            blocks[("Freq. smallest MSFE", lab)] = best
            blocks[("Freq. largest MSFE", lab)] = worst
        df = pd.DataFrame(blocks)
        order = ["Ratio of ave. MSFE", "Freq. beating benchmark", "Freq. smallest MSFE", "Freq. largest MSFE"]
        df = df[[c for o in order for c in df.columns if c[0] == o]]
        df.index = [method_label(m) for m in rows]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    # ---------------------------------------------------------------- Table 3
    def dm_tests(self, methods: Optional[Sequence[str]] = None, variance: str = "independent",
                 crit: float = 1.96, cond: str = "all") -> pd.DataFrame:
        """Table 3: panel DM statistic and counts of significant unit DM tests.

        Negative values mean the method is more accurate than the benchmark.
        """
        rows = [m for m in (methods or self.methods) if m != self.benchmark]
        e = self._err(cond)
        b = self._j(self.benchmark)
        out = {}
        for m in rows:
            D = e[:, :, self._j(m)] ** 2 - e[:, :, b] ** 2
            keep = (~np.isnan(D)).sum(axis=0) > 3
            D = D[:, keep]
            pdm = panel_dm_test(D, h=self.horizon, variance=variance)
            unit = np.array([dm_test(D[:, i], h=self.horizon).statistic for i in range(D.shape[1])])
            out[method_label(m)] = {
                "Panel DM": pdm.statistic, "p-value": pdm.pvalue,
                f"DM<-{crit:g}": int(np.sum(unit < -crit)),
                "insignificant": int(np.sum((unit > -crit) & (unit < crit))),
                f"DM>{crit:g}": int(np.sum(unit > crit)), "units": int(D.shape[1])}
        return pd.DataFrame(out)

    def unit_dm(self, method: str) -> pd.DataFrame:
        """Unit-by-unit DM and HLN statistics for one method against the benchmark."""
        e = self.errors
        D = e[:, :, self._j(method)] ** 2 - e[:, :, self._j(self.benchmark)] ** 2
        res = [dm_test(D[:, i], h=self.horizon) for i in range(D.shape[1])]
        return pd.DataFrame({"DM": [r.statistic for r in res], "p": [r.pvalue for r in res],
                             "HLN": [r.statistic_hln for r in res], "p_HLN": [r.pvalue_hln for r in res],
                             "n": [r.n for r in res]}, index=self.panel.units)

    # -------------------------------------------------------------- reports
    def summary(self, methods=None, best_worst_among=None, scale: float = 1.0, variance="independent"):
        from ._tables import evaluation_summary
        return evaluation_summary(self, methods, best_worst_among, scale, variance)

    def to_latex(self, which: str = "msfe", **kw) -> str:
        from ._tables import evaluation_latex
        return evaluation_latex(self, which, **kw)

    def plot_density(self, methods=None, cond="all", **kw):
        from ._plots import plot_msfe_density
        return plot_msfe_density(self, methods, cond, **kw)

    def plot_heatmap(self, methods=None, cond="all", **kw):
        from ._plots import plot_relative_heatmap
        return plot_relative_heatmap(self, methods, cond, **kw)

    def plot_cumulative(self, methods=None, **kw):
        from ._plots import plot_cumulative_loss
        return plot_cumulative_loss(self, methods, **kw)

    def __repr__(self) -> str:
        return (f"EvaluationResults({len(self.origins)} origins {self.target_labels[0]}..{self.target_labels[-1]}, "
                f"{self.scheme} window={self.window}, h={self.horizon}, methods={len(self.methods)})")


class _quiet:
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._c.__exit__(*a)
