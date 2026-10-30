"""Panel data container: builds y_it and w_it = (1, x_it')' from a long DataFrame.

The paper's convention is that w_it holds predictors known at t - h. The
container builds them from raw variables with explicit lags:

* ``y_lags``  lags of the dependent variable (e.g. [1] for a panel AR(1));
* ``x``       other predictors, ``{name: lags}``; lag 0 is a contemporaneous
              (conditioning / scenario) variable whose future values must be
              supplied when forecasting;
* ``trend``   a deterministic linear trend.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["PanelData", "wide_to_long"]

Lags = Union[int, Sequence[int]]


def wide_to_long(df: pd.DataFrame, time: str, units: Sequence[str], value_name: str = "y",
                 unit_name: str = "unit", common: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Reshape a wide table (one column per unit) into the long format.

    Parameters
    ----------
    df : DataFrame
        One row per period; ``units`` are the columns holding the outcome of
        each unit; ``common`` are variables shared by all units (prices,
        indices, dummies) and are repeated for every unit.
    time : str
        Name of the period column.

    Examples
    --------
    >>> import pandas as pd
    >>> from forecastpanel import wide_to_long
    >>> wide = pd.DataFrame({"year": [2020, 2021], "A": [1., 2.], "B": [3., 4.], "p": [9., 8.]})
    >>> wide_to_long(wide, "year", ["A", "B"], "exports", common=["p"]).shape
    (4, 4)
    """
    common = list(common or [])
    missing = [c for c in [time, *units, *common] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    long = df.melt(id_vars=[time, *common], value_vars=list(units),
                   var_name=unit_name, value_name=value_name)
    return long[[unit_name, time, value_name, *common]].sort_values([unit_name, time]).reset_index(drop=True)


def _as_lags(v: Lags) -> List[int]:
    if isinstance(v, (int, np.integer)):
        return [int(v)]
    out = [int(a) for a in v]
    if any(a < 0 for a in out):
        raise ValueError("Lags must be >= 0")
    return out


@dataclass(frozen=True)
class Column:
    kind: str           # 'const' | 'y' | 'x' | 'trend'
    name: str = ""
    lag: int = 0

    @property
    def label(self) -> str:
        if self.kind == "const":
            return "const"
        if self.kind == "trend":
            return "trend"
        if self.lag == 0:
            return self.name
        return f"{self.name}(-{self.lag})"


class PanelData:
    """Balanced-in-time panel with an explicit predictor specification.

    Parameters
    ----------
    data : DataFrame
        Long format: one row per (unit, time).
    unit, time, y : str
        Column names of the unit identifier, the period and the outcome.
    y_lags : int or list of int, default [1]
        Lags of ``y`` used as predictors (``[]`` for none).
    x : list of str or dict, optional
        Other predictors. A list means lag 1 for each; a dict maps a name to a
        lag or list of lags; lag 0 makes the variable contemporaneous.
    exog : list of str, optional
        Shorthand for contemporaneous (lag 0) predictors, e.g. scenario dummies.
    trend : bool, default False
        Add a linear time trend (1, 2, ..., in the order of the periods).

    Examples
    --------
    >>> from forecastpanel import PanelData, load_house_prices
    >>> df = load_house_prices(with_neighbors_only=True)
    >>> p = PanelData(df, "msa", "quarter", "dhp", y_lags=[1], x={"dhp_us": 1})
    >>> p.N, p.labels
    (362, ['const', 'dhp(-1)', 'dhp_us(-1)'])
    """

    def __init__(self, data: pd.DataFrame, unit: str, time: str, y: str,
                 y_lags: Lags = (1,), x: Optional[Union[Sequence[str], Dict[str, Lags]]] = None,
                 exog: Optional[Sequence[str]] = None, trend: bool = False):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame in long format")
        for c in (unit, time, y):
            if c not in data.columns:
                raise KeyError(f"Column '{c}' not found in data")
        self.unit_col, self.time_col, self.y_name = unit, time, y
        xspec: Dict[str, List[int]] = {}
        if x is not None:
            if isinstance(x, dict):
                xspec = {k: _as_lags(v) for k, v in x.items()}
            else:
                xspec = {k: [1] for k in x}
        for e in exog or []:
            xspec.setdefault(e, [])
            if 0 not in xspec[e]:
                xspec[e].append(0)
        for k in xspec:
            if k not in data.columns:
                raise KeyError(f"Predictor '{k}' not found in data")
            if k == y:
                raise ValueError("Use y_lags for lags of the dependent variable")
        self.y_lags = _as_lags(y_lags) if y_lags is not None else []
        if 0 in self.y_lags:
            raise ValueError("y_lags must be >= 1")
        self.x_spec = xspec
        self.trend = bool(trend)

        d = data.copy()
        if d.duplicated([unit, time]).any():
            raise ValueError("Duplicate (unit, time) rows in data")
        self.units = list(pd.unique(d[unit]))
        self.times = list(np.sort(pd.unique(d[time])))
        idx = pd.MultiIndex.from_product([self.units, self.times], names=[unit, time])
        d = d.set_index([unit, time]).reindex(idx)
        self.N, self.Tall = len(self.units), len(self.times)
        self._y = d[y].to_numpy(float).reshape(self.N, self.Tall)
        self._x = {k: d[k].to_numpy(float).reshape(self.N, self.Tall) for k in xspec}

        cols = [Column("const")]
        cols += [Column("y", y, L) for L in self.y_lags]
        for k, lags in xspec.items():
            cols += [Column("x", k, L) for L in sorted(lags)]
        if self.trend:
            cols.append(Column("trend"))
        self.columns: List[Column] = cols
        self.max_lag = max([c.lag for c in cols] + [0])

    # ------------------------------------------------------------------ props
    @property
    def K(self) -> int:
        return len(self.columns)

    @property
    def labels(self) -> List[str]:
        return [c.label for c in self.columns]

    @property
    def contemporaneous(self) -> List[str]:
        return [c.name for c in self.columns if c.kind == "x" and c.lag == 0]

    @property
    def has_ylag(self) -> bool:
        return bool(self.y_lags)

    def __repr__(self) -> str:
        return (f"PanelData(N={self.N}, periods={self.Tall} [{self.times[0]}..{self.times[-1]}], "
                f"y='{self.y_name}', w_it={self.labels})")

    # ------------------------------------------------------------ design rows
    def _row(self, t: int, y: np.ndarray, x: Dict[str, np.ndarray]) -> np.ndarray:
        """w_{i,t} for all units at (0-based) period index t from arrays y, x."""
        out = np.empty((self.N, self.K))
        for j, c in enumerate(self.columns):
            if c.kind == "const":
                out[:, j] = 1.0
            elif c.kind == "trend":
                out[:, j] = t + 1.0
            else:
                src = y if c.kind == "y" else x[c.name]
                s = t - c.lag
                out[:, j] = src[:, s] if 0 <= s < src.shape[1] else np.nan
        return out

    def design(self) -> Tuple[np.ndarray, np.ndarray]:
        """Full outcome (N, Tall) and design (N, Tall, K) arrays; NaN where lags are missing."""
        W = np.stack([self._row(t, self._y, self._x) for t in range(self.Tall)], axis=1)
        return self._y.copy(), W

    def window(self, end: int, length: Optional[int] = None):
        """Estimation window ending at period index ``end`` (inclusive).

        Returns ``Y (N,T), W (N,T,K), t_index`` with t_index the period indices.
        With ``length=None`` the window starts at the first period where all
        lags are available (expanding scheme).
        """
        Yall, Wall = self.design()
        start = self.max_lag if length is None else end - length + 1
        if start < self.max_lag:
            raise ValueError(f"Window starts at period {start} but the first {self.max_lag} "
                             "periods are needed for lags.")
        t_idx = np.arange(start, end + 1)
        return Yall[:, t_idx], Wall[:, t_idx], t_idx

    def complete_units(self, Y: np.ndarray, W: np.ndarray, *extra: np.ndarray) -> np.ndarray:
        """Boolean mask of units with no missing value in the window (and in ``extra``)."""
        ok = ~np.isnan(Y).any(axis=1) & ~np.isnan(W).any(axis=(1, 2))
        for e in extra:
            e = np.asarray(e)
            ok &= ~np.isnan(e.reshape(e.shape[0], -1)).any(axis=1)
        return ok

    # ----------------------------------------------------------- future paths
    def future_labels(self, H: int) -> list:
        t = self.times
        if all(isinstance(v, (int, np.integer)) for v in t) and len(t) > 1:
            step = int(np.median(np.diff(t)))
            return [int(t[-1]) + step * (h + 1) for h in range(H)]
        if all(isinstance(v, str) for v in t):
            try:   # period strings such as '2023Q1' or '2022-12'
                last = pd.Period(t[-1])
                if str(pd.Period(t[0], freq=last.freq)) == t[0]:
                    return [str(last + h + 1) for h in range(H)]
            except (TypeError, ValueError):
                pass
        try:
            ser = pd.Index(t)
            freq = pd.infer_freq(pd.DatetimeIndex(ser))
            if freq is not None:
                return list(pd.date_range(ser[-1], periods=H + 1, freq=freq)[1:])
        except (TypeError, ValueError):
            pass
        return [f"T+{h + 1}" for h in range(H)]

    def extend(self, H: int, future: Optional[pd.DataFrame] = None,
               fill: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Predictor arrays extended by H future periods.

        ``future`` is a long (unit, time, vars) or time-only (time, vars)
        DataFrame with values of the x variables in the forecast periods.
        Missing future values that a forecast needs raise an error unless
        ``fill`` is 'last' (hold the last observed value) or 'mean' (unit mean).
        """
        labels = self.future_labels(H)
        xe = {k: np.concatenate([v, np.full((self.N, H), np.nan)], axis=1) for k, v in self._x.items()}
        if future is not None and len(xe):
            f = future.copy()
            tc = self.time_col
            if tc not in f.columns:
                raise KeyError(f"future must contain the time column '{tc}'")
            for h, lab in enumerate(labels):
                rows = f[f[tc] == lab]
                if rows.empty:
                    continue
                for k in xe:
                    if k not in rows.columns:
                        continue
                    if self.unit_col in rows.columns:
                        s = rows.set_index(self.unit_col)[k].reindex(self.units)
                        xe[k][:, self.Tall + h] = s.to_numpy(float)
                    else:
                        xe[k][:, self.Tall + h] = float(rows[k].iloc[0])
        if fill is not None:
            for k, arr in xe.items():
                last = arr[:, :self.Tall]
                if fill == "last":
                    base = pd.DataFrame(last.T).ffill().iloc[-1].to_numpy()
                elif fill == "mean":
                    base = np.nanmean(last, axis=1)
                else:
                    raise ValueError("fill must be None, 'last' or 'mean'")
                fut = arr[:, self.Tall:]
                miss = np.isnan(fut)
                fut[miss] = np.broadcast_to(base[:, None], fut.shape)[miss]
        return xe

    def path_regressors(self, theta: np.ndarray, H: int, x_ext: Dict[str, np.ndarray],
                        y_hist: Optional[np.ndarray] = None, add: Optional[np.ndarray] = None,
                        shocks: Optional[np.ndarray] = None):
        """Iterated forecasts y_{T+1..T+H} with unit coefficients ``theta`` (N, K).

        Lags of y beyond the sample are replaced by the method's own forecasts;
        ``add`` (N, H) adds a per-horizon shift (RE-AR(1) correction); ``shocks``
        (N, H) adds innovations (bootstrap predictive paths).
        """
        y = self._y if y_hist is None else y_hist
        ye = np.concatenate([y, np.full((self.N, H), np.nan)], axis=1)
        T0 = y.shape[1]
        out = np.empty((self.N, H))
        wpath = np.empty((self.N, H, self.K))
        for h in range(H):
            w = self._row(T0 + h, ye, x_ext)
            if np.isnan(w).any():
                bad = sorted({c.label for j, c in enumerate(self.columns) if np.isnan(w[:, j]).any()})
                raise ValueError(
                    f"Missing predictor values for forecast step {h + 1}: {bad}. Supply them in "
                    "`future=` (scenario values) or pass fill='last'.")
            f = (w * theta).sum(axis=1)
            if add is not None:
                f = f + add[:, h]
            if shocks is not None:
                f = f + shocks[:, h]
            ye[:, T0 + h] = f
            out[:, h] = f
            wpath[:, h] = w
        return out, wpath
