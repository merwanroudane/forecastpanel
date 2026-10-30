"""Tests of equal predictive accuracy.

* ``dm_test``       Diebold and Mariano (1995) with the small-sample correction
                    of Harvey, Leybourne and Newbold (1997)  (``modifiedDieboldMariano.m``)
* ``panel_dm_test`` panel DM test of Pesaran, Pick and Pranovich (2013, Appendix 2)
                    (``PanelDieboldMarianoPPP_matlab.m``)

Sign convention of the paper: the loss differential is d = loss(method) -
loss(benchmark), so negative statistics mean the method is MORE accurate than
the benchmark.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

__all__ = ["dm_test", "panel_dm_test", "DMResult"]


@dataclass
class DMResult:
    """Outcome of a (panel) Diebold-Mariano test."""

    statistic: float
    pvalue: float
    statistic_hln: float = np.nan
    pvalue_hln: float = np.nan
    n: int = 0
    mean_loss_diff: float = np.nan
    method: str = "DM"

    def __repr__(self) -> str:
        s = f"{self.method}: stat={self.statistic:.3f}, p={self.pvalue:.4f}, n={self.n}"
        if np.isfinite(self.statistic_hln):
            s += f" | HLN stat={self.statistic_hln:.3f}, p={self.pvalue_hln:.4f}"
        return s


def dm_test(d, h: int = 1, *, loss_a=None, loss_b=None) -> DMResult:
    """Diebold-Mariano (1995) test and its Harvey-Leybourne-Newbold (1997) version.

    Parameters
    ----------
    d : array_like
        Loss differential d_t = g(e_{a,t}) - g(e_{b,t}); NaNs are dropped.
        Alternatively pass ``loss_a`` and ``loss_b`` and leave ``d=None``.
    h : int
        Forecast horizon; autocovariances up to lag h-1 enter the long-run
        variance, with a Bartlett fallback when it is negative (as in the code).

    Returns
    -------
    DMResult
        ``statistic`` is the DM statistic used in Table 3 of the paper; the
        HLN-corrected statistic is compared with Student-t(n-1).

    Examples
    --------
    >>> import numpy as np
    >>> from forecastpanel import dm_test
    >>> rng = np.random.default_rng(1)
    >>> r = dm_test(rng.normal(-0.3, 1, 200))
    >>> bool(r.statistic < 0)
    True
    """
    if d is None:
        d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = np.asarray(d, dtype=float).ravel()
    d = d[~np.isnan(d)]
    n = d.size
    if n < 4:
        return DMResult(np.nan, np.nan, np.nan, np.nan, n, np.nan)
    dm = d - d.mean()
    g = [dm[i:] @ dm[:-i] / n for i in range(1, h)]
    V = (np.var(d, ddof=1) + 2 * sum(g)) / n
    if V < 0:
        V = np.var(d, ddof=1)
        for i in range(1, int(n / 2) + 1):
            V += (1 / n) * (1 - i / (n / 2 + 1)) * (dm[i:] @ dm[:-i])
    s = d.mean() / np.sqrt(V)
    sstar = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * s
    p = 2 * stats.norm.sf(abs(s))
    pstar = 2 * stats.t.sf(abs(sstar), df=n - 1)
    return DMResult(float(s), float(p), float(sstar), float(pstar), n, float(d.mean()), "DM")


def panel_dm_test(D, h: int = 1, weights: Optional[np.ndarray] = None,
                  bandwidth: Optional[int] = None, variance: str = "independent",
                  min_obs: int = 4) -> DMResult:
    """Panel Diebold-Mariano test of Pesaran, Pick and Pranovich (2013).

    Parameters
    ----------
    D : array_like (T, N)
        Loss differentials, one column per unit (NaN allowed).
    h : int
        Forecast horizon; for h > 1 a Bartlett correction with ``bandwidth``
        (default h - 1) is added to each unit's variance.
    weights : array (N,), optional
        Aggregation weights, default 1/N.
    variance : {'independent', 'cross-section-average', 'ppp-legacy'}
        'independent' (default): DM = sum w_i dbar_i / sqrt(sum w_i^2 s_i^2 / T_i),
        PPP (2013) under cross-sectional independence; identical to the code for
        balanced panels.
        'cross-section-average': the variant described in footnote 17 of the
        paper - a Newey-West HAC t-test on the time series of cross-sectional
        average loss differentials, robust to cross-sectional dependence.
        'ppp-legacy': the unbalanced-panel branch of the replication code,
        which scales the variance by sum_j 1/T_j instead of 1/T_i (this
        inflates the variance by about N; it is what produced the CPI panel DM
        values of Table 3).  # PAPER vs CODE
    min_obs : int
        Units with fewer non-missing differentials are dropped (code: > 3).

    Returns
    -------
    DMResult (p-value from the standard normal).
    """
    D = np.asarray(D, dtype=float)
    if D.ndim == 1:
        D = D[:, None]
    keep = (~np.isnan(D)).sum(axis=0) >= min_obs
    D = D[:, keep]
    T, N = D.shape
    if N == 0:
        return DMResult(np.nan, np.nan, n=0, method="Panel DM")
    w = np.full(N, 1.0 / N) if weights is None else np.asarray(weights, float)[keep]
    sh = (h - 1) if bandwidth is None else bandwidth

    if variance == "cross-section-average":
        z = np.nanmean(D, axis=1)
        z = z[~np.isnan(z)]
        n = z.size
        zc = z - z.mean()
        L = sh if bandwidth is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        lrv = zc @ zc / n
        for j in range(1, L + 1):
            lrv += 2 * (1 - j / (L + 1)) * (zc[j:] @ zc[:-j]) / n
        stat = z.mean() / np.sqrt(lrv / n)
        return DMResult(float(stat), float(2 * stats.norm.sf(abs(stat))), n=n,
                        mean_loss_diff=float(z.mean()), method="Panel DM (CSA)")

    zbar = np.empty(N)
    s2 = np.empty(N)
    Ti = np.empty(N)
    for i in range(N):
        zi = D[:, i]
        zi = zi[~np.isnan(zi)]
        Ti[i] = zi.size
        zbar[i] = zi.mean()
        s2[i] = np.var(zi, ddof=1)
        if h > 1:
            zc = zi - zbar[i]
            add = sum((1 - s / (1 + sh)) * (zc[s:] @ zc[:-s]) for s in range(1, sh + 1))
            s2[i] += 2 / Ti[i] * add
    zb = float(w @ zbar)
    if variance == "independent":
        var = float((w ** 2 * s2 / Ti).sum())
    elif variance == "ppp-legacy":
        balanced = not np.isnan(D).any()
        Tall = np.full(N, T, dtype=float)
        var = float((w ** 2 * s2).sum() / T) if balanced else float((w ** 2 * s2).sum() * (1 / Tall).sum())
    else:
        raise ValueError("variance must be 'independent', 'cross-section-average' or 'ppp-legacy'")
    stat = zb / np.sqrt(var)
    return DMResult(float(stat), float(2 * stats.norm.sf(abs(stat))), n=int(Ti.sum()),
                    mean_loss_diff=zb, method="Panel DM")
