"""High-level interface: ``PanelForecaster`` -> ``ForecastResults``.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._engine import PanelFit, _resolve, method_label, panel_forecasts
from ._evaluation import evaluate_forecasts
from ._panel import PanelData

__all__ = ["PanelForecaster", "ForecastResults", "forecast_panel"]


class PanelForecaster:
    """Panel forecasting model of Pesaran, Pick and Timmermann (2026).

    Parameters
    ----------
    data : DataFrame
        Long-format panel (one row per unit and period).
    unit, time, y : str
        Columns with the unit identifier, the period and the target variable.
    y_lags : int or list, default [1]
        Lags of the target used as predictors.
    x : list or dict, optional
        Other predictors; a list means lag 1, a dict ``{name: lags}``; lag 0 is
        contemporaneous (its future values come from a scenario).
    exog : list, optional
        Contemporaneous predictors (scenario variables).
    trend : bool
        Linear time trend.
    methods : list of str, 'paper', 'all' or None
        Forecasting methods; ``None`` = the paper's eight + two equal-weight combinations.
    **options
        ``fe_weight`` ('code'/'paper'), ``unit_sigma`` ('ppt2022'/'ols'),
        ``omega_regularize`` (None/'ridge'/'diagonal'), ``hb_options`` (dict).

    Examples
    --------
    >>> import forecastpanel as fp
    >>> df = fp.load_house_prices(with_neighbors_only=True)
    >>> df = df[df["msa"].isin(df["msa"].unique()[:40])]
    >>> model = fp.PanelForecaster(df, "msa", "quarter", "dhp", y_lags=[1],
    ...                            x={"dhp_us": 1})
    >>> res = model.fit(window=40)
    >>> fc = res.forecast(horizon=2, fill="last")
    >>> fc.shape
    (80, 10)
    """

    def __init__(self, data: pd.DataFrame, unit: str, time: str, y: str, y_lags=(1,),
                 x=None, exog=None, trend: bool = False, methods=None, **options):
        self.panel = PanelData(data, unit, time, y, y_lags=y_lags, x=x, exog=exog, trend=trend)
        self.methods = _resolve(methods)
        self.options = options

    @classmethod
    def from_panel(cls, panel: PanelData, methods=None, **options) -> "PanelForecaster":
        obj = cls.__new__(cls)
        obj.panel, obj.methods, obj.options = panel, _resolve(methods), options
        return obj

    def fit(self, window: Optional[int] = None, end=None, future: Optional[pd.DataFrame] = None,
            seed: Optional[int] = None) -> "ForecastResults":
        """Estimate every method on the last ``window`` periods (default: all) up to ``end``.

        The combination weights depend on w_{i,T+1}. Lagged predictors come from
        the data; contemporaneous predictors of period T+1 come from the data,
        from ``future`` (scenario) or, failing both, their last observed value.
        """
        p = self.panel
        e = p.Tall - 1 if end is None else p.times.index(end)
        Y, W, t_idx = p.window(e, window)
        ok = p.complete_units(Y, W)
        if not ok.all():
            dropped = [p.units[i] for i in np.flatnonzero(~ok)]
            import warnings
            warnings.warn(f"Units with missing data in the window are dropped: {dropped}", stacklevel=2)
        x_ext = p.extend(max(0, e + 1 - (p.Tall - 1)), future=future, fill="last")
        yext = np.concatenate([p._y[:, :e + 1], np.full((p.N, 1), np.nan)], axis=1)
        wn = p._row(e + 1, yext, x_ext)
        rng = np.random.default_rng(seed)
        fit = panel_forecasts(Y[ok], W[ok], wn[ok], self.methods, rng=rng, **self.options)
        return ForecastResults(self, fit, ok, e, t_idx, Y, W)

    def evaluate(self, window: Optional[int] = 60, scheme: str = "rolling", n_forecasts=None,
                 first_origin=None, horizon: int = 1, **kw):
        """Pseudo out-of-sample evaluation; see :func:`evaluate_forecasts`."""
        opts = {**self.options, **kw}
        return evaluate_forecasts(self.panel, window=window, scheme=scheme, n_forecasts=n_forecasts,
                                  first_origin=first_origin, horizon=horizon, methods=self.methods, **opts)


def forecast_panel(data, unit, time, y, horizon: int = 1, window=None, future=None, **kw) -> pd.DataFrame:
    """One-call interface: fit and return the forecast table.

    Examples
    --------
    >>> import forecastpanel as fp
    >>> df = fp.load_house_prices(with_neighbors_only=True)
    >>> df = df[df["msa"].isin(df["msa"].unique()[:30])]
    >>> fp.forecast_panel(df, "msa", "quarter", "dhp", window=40).shape
    (30, 10)
    """
    model = PanelForecaster(data, unit, time, y, **kw)
    return model.fit(window=window).forecast(horizon=horizon, future=future)


class ForecastResults:
    """Fitted panel forecasting methods on one estimation window."""

    def __init__(self, model: PanelForecaster, fit: PanelFit, ok, end, t_idx, Y, W):
        self.model = model
        self.panel: PanelData = model.panel
        self.fit = fit
        self.methods = fit.methods
        self.units_used = [self.panel.units[i] for i in np.flatnonzero(ok)]
        self._ok = ok
        self.end = end
        self.t_idx = t_idx
        self._Y, self._W = Y[ok], W[ok]
        self._last_forecast: Optional[pd.DataFrame] = None
        self._last_intervals = None

    # ------------------------------------------------------------- basics
    @property
    def N(self) -> int:
        return self.fit.N

    @property
    def T(self) -> int:
        return self.fit.T

    @property
    def sample(self) -> str:
        t = self.panel.times
        return f"{t[self.t_idx[0]]}-{t[self.t_idx[-1]]}"

    def theta(self, method: str) -> pd.DataFrame:
        """Unit-specific coefficients theta_i of a method (rows: units)."""
        return pd.DataFrame(self.fit.theta[method], index=self.units_used, columns=self.panel.labels)

    @property
    def weights(self) -> Dict[str, object]:
        return self.fit.weights

    def unit_weights(self) -> pd.Series:
        """omega_i* of the unit-specific combination (weight on the individual forecast)."""
        if "comb_unit" not in self.fit.weights:
            raise KeyError("Include 'comb_unit' in methods")
        return pd.Series(self.fit.weights["comb_unit"].weights, index=self.units_used, name="omega_i")

    def eb_shrinkage(self) -> pd.DataFrame:
        """Diagonal of the EB weight matrix W_iT (eq. 45): weight on theta_hat_i per coefficient."""
        eb = self.fit.details["eb"]
        return pd.DataFrame(np.einsum("ikk->ik", eb.shrinkage), index=self.units_used, columns=self.panel.labels)

    # ----------------------------------------------------------- forecasts
    def _path_inputs(self, horizon, future, fill):
        # realised x inside the sample, scenario values beyond it
        p = self.panel
        H_out = max(0, self.end + horizon - (p.Tall - 1))
        x_ext = p.extend(H_out, future=future, fill=fill)
        return x_ext, p._y[:, :self.end + 1]

    def forecast(self, horizon: int = 1, future: Optional[pd.DataFrame] = None,
                 fill: Optional[str] = None, offsets: Optional[Dict] = None) -> pd.DataFrame:
        """Point forecasts for horizons 1..H for every method.

        Multi-step forecasts are iterated: lags of y beyond the sample are
        replaced by the method's own forecasts (EXTENSION - the paper studies
        h = 1). Contemporaneous predictors in the forecast periods come from
        ``future`` (a scenario); ``fill='last'`` holds missing values at their
        last observation.

        Parameters
        ----------
        horizon : int
        future : DataFrame, optional
            Scenario values of the predictors: columns ``time`` [, ``unit``] and
            the variable names.
        fill : {None, 'last', 'mean'}
        offsets : dict {unit: array of length H}, optional
            Additive adjustments to a unit's forecasts (e.g. new capacity that
            the data cannot identify), applied after estimation.

        Returns
        -------
        DataFrame indexed by (unit, period) with one column per method.
        """
        p = self.panel
        x_ext, y_hist = self._path_inputs(horizon, future, fill)
        idx_units = np.flatnonzero(self._ok)
        labels = [p.times[self.end + h + 1] if self.end + h + 1 < p.Tall else None for h in range(horizon)]
        fut = p.future_labels(horizon + self.end + 1 - p.Tall) if self.end + horizon >= p.Tall else []
        labels = [lab if lab is not None else fut[h - (p.Tall - 1 - self.end)] for h, lab in enumerate(labels)]
        out = {}
        for m in self.methods:
            th = np.full((p.N, p.K), np.nan)
            th[idx_units] = self.fit.theta[m]
            add = None
            if m == "re_ar1":
                ra = self.fit.details["re_ar1"]
                add = np.full((p.N, horizon), np.nan)
                th[idx_units, 0] = ra.theta_gls[0]
                add[idx_units] = np.column_stack([ra.correction(h + 1) for h in range(horizon)])
            if m == "hb" and horizon == 1:
                vals = np.full((p.N, 1), np.nan)
                vals[idx_units, 0] = self.fit.forecast["hb"]
            else:
                y_use = y_hist.copy()
                vals, _ = _paths(p, th, horizon, x_ext, y_use, add, idx_units)
            out[m] = vals[idx_units]
        rows = pd.MultiIndex.from_product([self.units_used, labels], names=[p.unit_col, p.time_col])
        df = pd.DataFrame({method_label(m): out[m].reshape(-1) for m in self.methods}, index=rows)
        if offsets:
            for u, adj in offsets.items():
                adj = np.broadcast_to(np.asarray(adj, float), (horizon,))
                df.loc[u] = df.loc[u].to_numpy() + adj[:, None]
        self._last_forecast = df
        return df

    def intervals(self, horizon: int = 1, future=None, fill=None, n_boot: int = 499,
                  level: float = 0.90, methods: Optional[Sequence[str]] = None,
                  seed: Optional[int] = None, design: str = "fixed") -> pd.DataFrame:
        """Bootstrap predictive intervals (EXTENSION, not in the paper).

        Residual bootstrap from the individual models. For each replication
        each unit's centred, dof-rescaled OLS residuals are resampled and
        y*_it = w_it'theta_hat_i + e*_it is formed, every method is re-estimated
        on y*, and an H-step predictive path is simulated from the observed
        history with resampled future shocks. Quantiles of the simulated paths
        give intervals reflecting estimation and innovation uncertainty.

        ``design='fixed'`` (default) keeps the observed regressors, including
        lagged y (fixed-design bootstrap; robust when some unit models are
        near-explosive). ``design='recursive'`` regenerates lagged y from the
        bootstrap sample; units whose individual lag polynomial is explosive
        are then generated with their empirical Bayes coefficients.

        Returns
        -------
        DataFrame indexed by (unit, period) with columns (method, 'lower'/'median'/'upper').
        """
        p = self.panel
        meths = [m for m in (methods or self.methods) if m != "hb"]
        rng = np.random.default_rng(seed)
        ind = self.fit.details["individual"]
        Y, W = self._Y, self._W
        N, T, K = W.shape
        res = ind.resid - ind.resid.mean(axis=1, keepdims=True)
        res *= np.sqrt(T / (T - K))
        idx_units = np.flatnonzero(self._ok)
        x_ext, y_hist = self._path_inputs(horizon, future, fill)
        ycols = [j for j, c in enumerate(p.columns) if c.kind == "y"]
        lags = [p.columns[j].lag for j in ycols]
        draws = {m: np.empty((n_boot, N, horizon)) for m in meths}
        wn_base = p._row(self.end + 1, np.concatenate([y_hist, np.full((p.N, 1), np.nan)], axis=1), x_ext)
        wn_base = wn_base[idx_units]
        if design not in ("fixed", "recursive"):
            raise ValueError("design must be 'fixed' or 'recursive'")
        theta_dgp = ind.theta.copy()
        if design == "recursive" and ycols:
            explosive = np.abs(ind.theta[:, ycols]).sum(axis=1) >= 1
            if explosive.any():
                from ._core import empirical_bayes
                try:
                    theta_dgp[explosive] = empirical_bayes(Y, W, ind, self.model.options.get(
                        "omega_regularize")).theta[explosive]
                except np.linalg.LinAlgError:
                    theta_dgp[explosive] = ind.theta.mean(axis=0)
        for b in range(n_boot):
            e = res[np.arange(N)[:, None], rng.integers(0, T, size=(N, T))]
            yfull = y_hist[idx_units].copy()
            if design == "fixed":
                Ws = W
                Ys = np.einsum("itk,ik->it", W, theta_dgp) + e
                wn = wn_base
            else:
                Ys = np.empty_like(Y)
                Ws = W.copy()
                t0 = yfull.shape[1] - T
                for t in range(T):
                    for j, L in zip(ycols, lags):
                        Ws[:, t, j] = yfull[:, t0 + t - L]
                    Ys[:, t] = (Ws[:, t] * theta_dgp).sum(axis=1) + e[:, t]
                    yfull[:, t0 + t] = Ys[:, t]
                wn = wn_base.copy()
                for j, L in zip(ycols, lags):
                    wn[:, j] = yfull[:, -L]
            try:
                fb = panel_forecasts(Ys, Ws, wn, meths, rng=rng, **self.model.options)
            except (np.linalg.LinAlgError, ValueError):
                for m in meths:
                    draws[m][b] = np.nan
                continue
            shocks = res[np.arange(N)[:, None], rng.integers(0, T, size=(N, horizon))]
            yh = np.full((p.N, y_hist.shape[1]), np.nan)
            yh[idx_units] = yfull
            for m in meths:
                th = np.full((p.N, p.K), np.nan)
                th[idx_units] = fb.theta[m]
                sh = np.zeros((p.N, horizon))
                sh[idx_units] = shocks
                vals, _ = _paths(p, th, horizon, x_ext, yh, None, idx_units, shocks=sh)
                draws[m][b] = vals[idx_units]
        a = (1 - level) / 2
        point = self.forecast(horizon, future, fill)
        cols = {}
        for m in meths:
            d = draws[m]
            q = np.nanquantile(d, [a, 0.5, 1 - a], axis=0)
            lab = method_label(m)
            cols[(lab, "forecast")] = point[lab].to_numpy()
            cols[(lab, "lower")] = q[0].reshape(-1)
            cols[(lab, "median")] = q[1].reshape(-1)
            cols[(lab, "upper")] = q[2].reshape(-1)
        out = pd.DataFrame(cols, index=point.index)
        out.columns = pd.MultiIndex.from_tuples(out.columns)
        out.attrs["level"] = level
        out.attrs["n_boot"] = n_boot
        self._last_intervals = out
        return out

    # ------------------------------------------------------- coefficients
    def coefficients(self) -> pd.DataFrame:
        """Coefficient table: pooled, FE and mean-group estimates with standard errors.

        Pooled and FE use cluster-robust (by unit) standard errors; the mean
        group uses the Pesaran-Smith (1995) estimator sqrt(diag(S)/N),
        S = (N-1)^-1 sum (theta_i - theta_MG)(theta_i - theta_MG)'.
        """
        from scipy import stats
        Y, W = self._Y, self._W
        N, T, K = W.shape
        labels = self.panel.labels
        rows = []
        # pooled
        WW = np.einsum("itk,itl->kl", W, W)
        tp = np.linalg.pinv(WW) @ np.einsum("itk,it->k", W, Y)
        u = Y - np.einsum("itk,k->it", W, tp)
        g = np.einsum("itk,it->ik", W, u)
        iWW = np.linalg.pinv(WW)
        V = N / (N - 1) * (N * T - 1) / (N * T - K) * iWW @ (g.T @ g) @ iWW
        rows.append(("Pooled", tp, np.sqrt(np.diag(V))))
        # FE
        fe = self.fit.details.get("fe")
        from ._core import fixed_effects, individual_ols
        fe = fe or fixed_effects(Y, W)
        X = W[:, :, 1:]
        Xd = X - X.mean(axis=1, keepdims=True)
        ud = (Y - Y.mean(axis=1, keepdims=True)) - np.einsum("itk,k->it", Xd, fe.beta)
        XX = np.einsum("itk,itl->kl", Xd, Xd)
        gx = np.einsum("itk,it->ik", Xd, ud)
        iXX = np.linalg.pinv(XX)
        Vf = N / (N - 1) * iXX @ (gx.T @ gx) @ iXX
        rows.append(("FE", np.r_[np.nan, fe.beta], np.r_[np.nan, np.sqrt(np.diag(Vf))]))
        ind = self.fit.details["individual"]
        mg = ind.theta.mean(axis=0)
        S = np.cov(ind.theta.T, ddof=1).reshape(K, K)
        rows.append(("Mean group", mg, np.sqrt(np.diag(S) / N)))
        recs = []
        for name, b, se in rows:
            for j, lab in enumerate(labels):
                t = b[j] / se[j] if np.isfinite(se[j]) and se[j] > 0 else np.nan
                pv = 2 * stats.t.sf(abs(t), df=N - 1) if np.isfinite(t) else np.nan
                recs.append({"estimator": name, "variable": lab, "coef": b[j], "se": se[j],
                             "t": t, "p": pv})
        return pd.DataFrame(recs).set_index(["estimator", "variable"])

    # ------------------------------------------------------------ outputs
    def summary(self):
        from ._tables import forecast_summary
        return forecast_summary(self)

    def plot_forecasts(self, units=None, methods=None, horizon=None, **kw):
        from ._plots import plot_forecast_paths
        return plot_forecast_paths(self, units, methods, horizon, **kw)

    def plot_coefficients(self, variable=None, **kw):
        from ._plots import plot_coefficient_dispersion
        return plot_coefficient_dispersion(self, variable, **kw)

    def plot_weights(self, **kw):
        from ._plots import plot_unit_weights
        return plot_unit_weights(self, **kw)

    def __repr__(self) -> str:
        return (f"ForecastResults(N={self.N}, T={self.T}, K={self.fit.K}, sample={self.sample}, "
                f"methods={[method_label(m) for m in self.methods]})")


def _paths(p: PanelData, th, horizon, x_ext, y_hist, add, idx_units, shocks=None):
    """Iterate forecasts for the units in ``idx_units`` (others stay NaN)."""
    sub = PanelData.__new__(PanelData)
    sub.__dict__.update(p.__dict__)
    sub.N = len(idx_units)
    xs = {k: v[idx_units] for k, v in x_ext.items()}
    vals, w = sub.path_regressors(th[idx_units], horizon, xs, y_hist=y_hist[idx_units],
                                  add=None if add is None else add[idx_units],
                                  shocks=None if shocks is None else shocks[idx_units])
    out = np.full((p.N, horizon), np.nan)
    out[idx_units] = vals
    return out, w
