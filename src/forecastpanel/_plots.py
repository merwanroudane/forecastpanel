"""Publication-quality figures. Every function returns ``(fig, ax)`` and never calls ``show``.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._colors import METHOD_COLORS, METHOD_LINESTYLES, resolve_cmap
from ._engine import method_label

JOURNAL_RC = {
    "figure.figsize": (6.5, 4.0), "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6,
}


def _rc():
    import matplotlib.pyplot as plt
    return plt.rc_context(JOURNAL_RC)


def _style(i: int):
    return METHOD_COLORS[i % len(METHOD_COLORS)], METHOD_LINESTYLES[i % len(METHOD_LINESTYLES)]


def save_figure(fig, stem: str, formats: Sequence[str] = ("pdf", "png")):
    """Save a figure as vector PDF and 300-dpi PNG: ``save_figure(fig, 'fig1')``."""
    for f in formats:
        fig.savefig(f"{stem}.{f}", dpi=300, bbox_inches="tight")


def _kde(y: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    # densityplotcalcs.m: average of normal pdfs centred on the data
    y = y[np.isfinite(y)]
    z = (grid[:, None] - y[None, :]) / bw
    return np.exp(-0.5 * z ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))


def plot_msfe_density(ev, methods: Optional[Sequence[str]] = None, cond: str = "all",
                      bandwidth: float = 0.03, xlim=(0.6, 1.4), ax=None, title=None):
    """Figure 1: densities of unit-level MSFE ratios (method / individual).

    Normal kernel; the replication code uses bandwidth 0.03 (the figure note
    of the paper says 0.04).
    """
    import matplotlib.pyplot as plt
    methods = methods or [m for m in ["pooled", "comb_pool", "comb_fe", "comb_unit", "eb"] if m in ev.methods]
    rel = ev.relative_msfe(cond)
    grid = np.linspace(xlim[0], xlim[1], 321)
    with _rc():
        fig, ax = (plt.subplots() if ax is None else (ax.figure, ax))
        for i, m in enumerate(methods):
            c, ls = _style(i)
            ax.plot(grid, _kde(rel[m].to_numpy(), grid, bandwidth), color=c, ls=ls, label=method_label(m))
        ax.axvline(1.0, color="0.3", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Ratio of MSFE (method / individual)")
        ax.set_ylabel("Density across units")
        names = {"all": "all forecasts", "kappa0": r"$\kappa_i=0$", "kappa1": r"$\kappa_i=\pm 1$"}
        ax.set_title(title or f"Distribution of MSFE ratios: {names.get(cond, cond)}")
        ax.legend()
    return fig, ax


def plot_relative_heatmap(ev, methods=None, cond="all", cmap="Parula", max_units: int = 60,
                          sort_by: str = "eb", ax=None):
    """Heatmap of unit MSFE ratios (units x methods), Parula colour scale by default."""
    import matplotlib.pyplot as plt
    methods = [m for m in (methods or ev.methods) if m != "individual"]
    rel = ev.relative_msfe(cond)[methods]
    if sort_by in rel.columns:
        rel = rel.sort_values(sort_by)
    if len(rel) > max_units:
        rel = rel.iloc[np.linspace(0, len(rel) - 1, max_units).round().astype(int)]
    from matplotlib.colors import LogNorm
    with _rc():
        h = max(4.0, 0.14 * len(rel) + 2.2)
        fig, ax = (plt.subplots(figsize=(7.0, h)) if ax is None else (ax.figure, ax))
        lr = np.abs(np.log(rel.to_numpy()))
        v = float(np.nanpercentile(lr[np.isfinite(lr)], 95)) if np.isfinite(lr).any() else 1.0
        v = max(v, 0.05)
        im = ax.imshow(rel.to_numpy(), aspect="auto", cmap=resolve_cmap(cmap),
                       norm=LogNorm(vmin=np.exp(-v), vmax=np.exp(v)))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([method_label(m) for m in methods], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(rel)))
        ax.set_yticklabels([str(u)[:18] for u in rel.index], fontsize=7)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, label="MSFE ratio to individual (log scale)")
        cb.ax.minorticks_off()
        ticks = [t for t in (0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0) if np.exp(-v) <= t <= np.exp(v)]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        ax.set_title("Unit-level relative MSFE (values below 1: method beats individual)")
        fig.tight_layout()
    return fig, ax


def plot_cumulative_loss(ev, methods=None, ax=None):
    """Cumulative sum over time of the cross-sectional average squared-error
    difference (benchmark minus method): rising lines mean the method gains."""
    import matplotlib.pyplot as plt
    methods = [m for m in (methods or ev.methods) if m != "individual"]
    b = ev.methods.index("individual")
    x = ev.target_labels
    with _rc():
        fig, ax = (plt.subplots() if ax is None else (ax.figure, ax))
        for i, m in enumerate(methods):
            j = ev.methods.index(m)
            d = np.nanmean(ev.errors[:, :, b] ** 2 - ev.errors[:, :, j] ** 2, axis=1)
            c, ls = _style(i)
            ax.plot(range(len(x)), np.nancumsum(d), color=c, ls=ls, label=method_label(m))
        ax.axhline(0, color="0.3", lw=0.8)
        step = max(1, len(x) // 8)
        ax.set_xticks(range(0, len(x), step))
        ax.set_xticklabels([str(v) for v in x[::step]], rotation=30, ha="right")
        ax.set_ylabel("Cumulative MSFE gain vs individual")
        ax.set_title("Cumulative out-of-sample gains")
        ax.legend(ncol=2)
    return fig, ax


def plot_forecast_paths(res, units=None, methods=None, horizon=None, history: int = 12,
                        intervals: bool = True, ncols: int = 3, sharey: bool = False, ylabel=None):
    """Small multiples: history of each unit and the forecast paths of the methods.

    Uses the last ``forecast()`` call (and ``intervals()`` bands for the first
    method when available).
    """
    import matplotlib.pyplot as plt
    p = res.panel
    fc = res._last_forecast if res._last_forecast is not None else res.forecast(horizon or 1)
    units = list(units or res.units_used)
    methods = methods or res.methods
    labs = [method_label(m) for m in methods]
    n = len(units)
    nr = int(np.ceil(n / ncols))
    times = p.times[:res.end + 1][-history:]
    with _rc():
        fig, axes = plt.subplots(nr, min(ncols, n), figsize=(3.2 * min(ncols, n), 2.4 * nr),
                                 squeeze=False, sharey=sharey)
        for k, u in enumerate(units):
            ax = axes[k // ncols][k % ncols]
            ui = p.units.index(u)
            hist = p._y[ui, :res.end + 1][-history:]
            xs_h = list(range(len(times)))
            ax.plot(xs_h, hist, color="0.15", lw=1.8, label="Observed")
            f = fc.loc[u]
            xs_f = list(range(len(times) - 1, len(times) + len(f)))
            if intervals and res._last_intervals is not None:
                iv = res._last_intervals.loc[u]
                lab0 = iv.columns.get_level_values(0)[0]
                ax.fill_between(xs_f[1:], iv[(lab0, "lower")], iv[(lab0, "upper")], color=METHOD_COLORS[0],
                                alpha=0.18, lw=0, label=f"{int(res._last_intervals.attrs['level'] * 100)}% band ({lab0})")
            for i, lab in enumerate(labs):
                c, ls = _style(i)
                ax.plot(xs_f, np.r_[hist[-1], f[lab].to_numpy()], color=c, ls=ls, lw=1.3, label=lab)
            ticks = times + list(f.index)
            step = max(1, len(ticks) // 5)
            ax.set_xticks(range(0, len(ticks), step))
            ax.set_xticklabels([str(t) for t in ticks[::step]], fontsize=7, rotation=30)
            ax.axvline(len(times) - 1, color="0.5", lw=0.7, ls=":")
            ax.set_title(str(u), fontsize=9.5)
            if ylabel and k % ncols == 0:
                ax.set_ylabel(ylabel)
        for k in range(n, nr * ncols):
            axes[k // ncols][k % ncols].axis("off")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), bbox_to_anchor=(0.5, 0.0))
    return fig, axes


def plot_coefficient_dispersion(res, variable=None, methods=("individual", "eb"), ax=None):
    """Caterpillar plot of unit coefficients: OLS vs empirical Bayes shrinkage,
    with pooled and mean-group values as reference lines."""
    import matplotlib.pyplot as plt
    labels = res.panel.labels
    j = labels.index(variable) if variable else (1 if len(labels) > 1 else 0)
    with _rc():
        fig, ax = (plt.subplots() if ax is None else (ax.figure, ax))
        ind = res.fit.theta["individual"][:, j]
        order = np.argsort(ind)
        for i, m in enumerate([m for m in methods if m in res.fit.theta]):
            c, _ = _style(i)
            ax.plot(range(len(order)), res.fit.theta[m][order, j], "o" if i == 0 else "s", ms=3.5,
                    color=c, label=method_label(m), alpha=0.85)
        mg = ind.mean()
        ax.axhline(mg, color="0.25", lw=1, ls="--", label="Mean group")
        if "pooled" in res.fit.theta:
            ax.axhline(res.fit.theta["pooled"][0, j], color=METHOD_COLORS[1], lw=1, ls="-.", label="Pooled")
        ax.set_xlabel("Units (sorted by individual estimate)")
        ax.set_ylabel(f"Coefficient on {labels[j]}")
        ax.set_title("Parameter heterogeneity and Bayesian shrinkage")
        ax.legend()
    return fig, ax


def plot_unit_weights(res, ax=None):
    """Histogram of the unit-specific weights omega_i* with the common weights marked."""
    import matplotlib.pyplot as plt
    w = res.unit_weights()
    with _rc():
        fig, ax = (plt.subplots() if ax is None else (ax.figure, ax))
        ax.hist(w.to_numpy(), bins=min(30, max(8, len(w) // 4)), range=(0, 1), color=METHOD_COLORS[0],
                alpha=0.75, edgecolor="white")
        for i, (k, lab) in enumerate([("comb_pool", "Comb. (pool)"), ("comb_fe", "Comb. (FE)")]):
            if k in res.weights:
                ax.axvline(res.weights[k].weight, color=METHOD_COLORS[i + 1], ls=["--", "-."][i], lw=1.4,
                           label=f"{lab}: {res.weights[k].weight:.3f}")
        ax.set_xlabel(r"Weight on the individual forecast $\omega_i^*$")
        ax.set_ylabel("Number of units")
        ax.set_title("Estimated combination weights")
        ax.legend()
    return fig, ax


def plot_monte_carlo(table: pd.DataFrame, methods=None, ax=None):
    """rMSFE against T for each method from a ``monte_carlo`` table."""
    import matplotlib.pyplot as plt
    methods = methods or [c for c in table.columns if c not in ("T", "N", "Individual")]
    with _rc():
        fig, ax = (plt.subplots() if ax is None else (ax.figure, ax))
        for i, m in enumerate(methods):
            c, ls = _style(i)
            ax.plot(table["T"], table[m], marker="o", ms=4, color=c, ls=ls, label=m)
        ax.axhline(1, color="0.3", lw=0.8)
        ax.set_xlabel("Time dimension T")
        ax.set_ylabel("MSFE relative to individual")
        ax.set_title("Monte Carlo: relative MSFE")
        ax.legend(ncol=2)
    return fig, ax
