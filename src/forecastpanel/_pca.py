"""Principal components of a panel with missing values (``principalComponentsNaN.m``).

EM algorithm in the spirit of Stock and Watson (2002): standardise each
series, fill the gaps with the common-component fit and iterate.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import numpy as np

__all__ = ["principal_components"]


def principal_components(X: np.ndarray, r: int = 1, tol: float = 1e-6, max_iter: int = 10000,
                         standardize: bool = True) -> np.ndarray:
    """First ``r`` principal-component factors of a (T x N) array with NaNs.

    Rows or columns that are entirely missing are dropped (their factor rows
    are NaN). Returns a (T x r) array.

    Examples
    --------
    >>> import numpy as np
    >>> from forecastpanel import principal_components
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=50)
    >>> X = np.outer(f, rng.normal(size=20)) + 0.1 * rng.normal(size=(50, 20))
    >>> X[3, 4] = np.nan
    >>> F = principal_components(X, 1)
    >>> bool(abs(np.corrcoef(F[:, 0], f)[0, 1]) > 0.99)
    True
    """
    X = np.asarray(X, dtype=float).copy()
    if standardize:
        X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0, ddof=1)
    T0 = X.shape[0]
    rows = ~np.all(np.isnan(X), axis=1)
    X = X[rows]
    X = X[:, ~np.all(np.isnan(X), axis=0)]
    miss = np.isnan(X)
    Z = np.where(miss, 0.0, X)
    F = np.zeros((X.shape[0], r))
    cc = tol / (X.shape[0] * r)
    for _ in range(max_iter):
        C = np.cov(Z, rowvar=False)
        vals, vecs = np.linalg.eigh(C)
        Fn = Z @ vecs[:, -r:][:, ::-1]
        lam = np.linalg.lstsq(Fn, Z, rcond=None)[0]
        Z = np.where(miss, Fn @ lam, X)
        if np.abs(F - Fn).sum() < cc:
            F = Fn
            break
        F = Fn
    out = np.full((T0, r), np.nan)
    out[rows] = F
    return out
