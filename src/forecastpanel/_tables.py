"""Journal-style tables: one DataFrame source, rendered as text, LaTeX and Markdown.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._engine import method_label

WIDTH = 92
REF = "Pesaran, Pick and Timmermann (2026, QE)"


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _f(v, fmt="{:.3f}", na="--"):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return na
    if fmt == "{:.3f}" and abs(v) >= 1000:
        return f"{v:.4g}" if abs(v) >= 1e5 else f"{v:.1f}"
    return fmt.format(v)


class TextReport:
    """String-like report whose ``repr`` is the table (prints well in Jupyter)."""

    def __init__(self, text: str, latex: str = "", frames: Optional[dict] = None):
        self.text, self.latex, self.frames = text, latex, frames or {}

    def __str__(self) -> str:
        return self.text

    __repr__ = __str__

    def as_text(self) -> str:
        return self.text

    def as_latex(self) -> str:
        return self.latex

    def _repr_html_(self) -> str:
        return "<pre style='font-size:12px'>" + self.text.replace("<", "&lt;") + "</pre>"


def _kv_block(left, right) -> list:
    out = []
    for (a, b), (c, d) in zip(left, right):
        out.append(f"{a:<20}: {str(b):<24}{c:<20}: {str(d):>22}")
    return out


# ---------------------------------------------------------------------------
# ForecastResults.summary
# ---------------------------------------------------------------------------
def forecast_summary(res) -> TextReport:
    p = res.panel
    fit = res.fit
    L = ["=" * WIDTH, f"{'Panel forecasts  -  ' + REF:^{WIDTH}}", "=" * WIDTH]
    L += _kv_block(
        [("Dependent variable", p.y_name), ("Units (N)", res.N), ("Periods (T)", res.T),
         ("Sample", res.sample)],
        [("Regressors (K)", fit.K), ("Units dropped", p.N - res.N),
         ("Contemporaneous", len(p.contemporaneous)),
         ("Methods", len(res.methods))])
    wline = "w_it = (" + ", ".join(p.labels) + ")"
    while wline:
        L.append(wline[:WIDTH])
        wline = ("       " + wline[WIDTH:]) if len(wline) > WIDTH else ""
    L.append("-" * WIDTH)
    L.append("Combination weights (weight on the INDIVIDUAL forecast)")
    L.append(f"{'':<28}{'weight':>10}{'unclipped':>12}{'Delta':>11}{'h':>11}{'psi':>11}   eq.")
    L.append("-" * WIDTH)
    w = fit.weights
    if "comb_pool" in w:
        c = w["comb_pool"]
        L.append(f"{'Comb. (pool)':<28}{c.weight:>10.4f}{c.raw:>12.4f}{c.components['Delta']:>11.4f}"
                 f"{c.components['h']:>11.4f}{c.components['psi']:>11.4f}   (32)")
    if "comb_fe" in w:
        c = w["comb_fe"]
        L.append(f"{'Comb. (FE)':<28}{c.weight:>10.4f}{c.raw:>12.4f}{c.components['Delta']:>11.4f}"
                 f"{c.components['h']:>11.4f}{c.components['psi']:>11.4f}   (39)")
    for key in ("comb_unit", "comb_unit_ba"):
        if key in w:
            u = w[key].weights
            L.append(f"{method_label(key) + ' min/med/max':<28}{np.min(u):>10.4f}{np.median(u):>12.4f}"
                     f"{np.max(u):>11.4f}{'':>11}{'':>11}   (46)")
    for key in ("stein", "mallows"):
        if key in w:
            L.append(f"{method_label(key) + ' (1 - weight)':<28}{1 - w[key].weight:>10.4f}{1 - w[key].raw:>12.4f}")
    if "eq_pool" in fit.methods or "eq_fe" in fit.methods:
        L.append(f"{'Equal weights':<28}{0.5:>10.4f}")
    if "eb" in fit.details:
        sh = np.einsum("ikk->ik", fit.details["eb"].shrinkage)
        L.append(f"{'Emp. Bayes W_iT (mean diag)':<28}{sh.mean():>10.4f}{'':>12}{'':>11}{'':>11}{'':>11}   (45)")
    if "re" in fit.details:
        r = fit.details["re"]
        L.append(f"{'RE variance comps s2_a/s2_e':<28}{r.sigma2_alpha:>10.4f}{r.sigma2_eps:>12.4f}")
    L.append("-" * WIDTH)
    coef = res.coefficients()
    L.append(f"{'Coefficients':<14}{'':>20}{'Coef.':>12}{'Std.Err.':>11}{'t':>9}{'P>|t|':>9}")
    L.append("-" * WIDTH)
    for (est, var), r in coef.iterrows():
        if not np.isfinite(r["coef"]):
            continue
        L.append(f"{est:<14}{var[:20]:>20}{r['coef']:>12.4f}{_f(r['se'], '{:.4f}'):>11}"
                 f"{_f(r['t']):>9}{_f(r['p']):>9} {_stars(r['p'])}")
    L.append("=" * WIDTH)
    L.append("Significance: *** p<0.01, ** p<0.05, * p<0.10. Pooled/FE: cluster-robust (unit) SEs;")
    L.append("mean group: Pesaran-Smith (1995) SEs. Weights clipped to [0, 1] as in the replication code.")
    L.append("Delta: heterogeneity cost (40); h: estimation-error cost (41); psi: correlated-heterogeneity term (43).")
    text = "\n".join(L)
    return TextReport(text, coef.to_latex(float_format="%.4f", na_rep="--"), {"coefficients": coef})


# ---------------------------------------------------------------------------
# EvaluationResults.summary / latex
# ---------------------------------------------------------------------------
def evaluation_summary(ev, methods=None, best_worst_among=None, scale=1.0, variance="independent") -> TextReport:
    tab = ev.msfe_table(methods, best_worst_among, scale=scale)
    dm = ev.dm_tests([m for m in (methods or ev.methods)], variance=variance)
    conds = ["all", "kappa_i = 0", "kappa_i = +/-1"]
    blocks = ["Ratio of ave. MSFE", "Freq. beating benchmark", "Freq. smallest MSFE", "Freq. largest MSFE"]
    cw = 8
    W = 18 + 4 * 3 * cw
    L = ["=" * W, f"{'Out-of-sample forecast evaluation  -  ' + REF:^{W}}", "=" * W]
    L.append(f"Target: {ev.panel.y_name}   units: {ev.panel.N}   forecasts: {len(ev.origins)} "
             f"({ev.target_labels[0]} - {ev.target_labels[-1]})   {ev.scheme} window: {ev.window}   h = {ev.horizon}")
    L.append("-" * W)
    L.append(f"{'':<18}" + "".join(f"{b[:22]:^{3 * cw}}" for b in blocks))
    L.append(f"{'':<18}" + "".join(f"{c:>{cw}}" for c in ["all", "k=0", "k=+-1"] * 4))
    L.append("-" * W)
    for name, row in tab.iterrows():
        vals = [row[(b, c)] for b in blocks for c in conds]
        L.append(f"{name[:18]:<18}" + "".join(f"{_f(v):>{cw}}" for v in vals))
    L.append("-" * W)
    L.append("Diebold-Mariano tests against the individual forecasts (negative: method more accurate)")
    L.append(f"{'':<18}" + "".join(f"{c[:11]:>12}" for c in dm.columns))
    L.append(f"{'Panel DM':<18}" + "".join(f"{v:>12.2f}" for v in dm.loc['Panel DM']))
    L.append(f"{'  p-value':<18}" + "".join(f"{v:>12.3f}" for v in dm.loc['p-value']))
    cl = [i for i in dm.index if i.startswith("DM<")][0]
    cu = [i for i in dm.index if i.startswith("DM>")][0]
    L.append(f"{cl + ' / ' + cu:<18}" + "".join(f"{f'{int(a)}/{int(b)}':>12}" for a, b in zip(dm.loc[cl], dm.loc[cu])))
    L.append("=" * W)
    L.append("First row: average MSFE of the individual (benchmark) forecasts" + (f" x {scale:g}" if scale != 1 else "")
             + "; other rows: ratio to it.")
    L.append(f"kappa_i = 0 (+/-1): forecasts with d_i,T+1 = theta_i'w_i,T+1 within {ev.kappa_c:g} sd of its window mean (+/- 1 sd).")
    L.append("Panel DM: Pesaran, Pick and Pranovich (2013); unit DM: Diebold and Mariano (1995).")
    return TextReport("\n".join(L), evaluation_latex(ev, "msfe", methods=methods, scale=scale),
                      {"msfe": tab, "dm": dm})


def evaluation_latex(ev, which="msfe", methods=None, scale=1.0, best_worst_among=None,
                     variance="independent", caption=None, label=None) -> str:
    if which == "msfe":
        tab = ev.msfe_table(methods, best_worst_among, scale=scale)
        blocks = ["Ratio of ave. MSFE", "Freq. beating benchmark", "Freq. smallest MSFE", "Freq. largest MSFE"]
        conds = ["all", "kappa_i = 0", "kappa_i = +/-1"]
        head1 = " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{b}}}" for b in blocks) + r" \\"
        cm = "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(4))
        head2 = " & " + " & ".join([r"all & $\kappa_i=0$ & $\kappa_i=\pm1$"] * 4) + r" \\"
        body = []
        for name, row in tab.iterrows():
            vals = [row[(b, c)] for b in blocks for c in conds]
            body.append(_esc(name) + " & " + " & ".join(_f(v) for v in vals) + r" \\")
        cap = caption or f"Out-of-sample forecast accuracy: {_esc(ev.panel.y_name)}"
        return "\n".join([
            "% requires \\usepackage{booktabs}",
            r"\begin{table}[htbp]\centering\footnotesize",
            rf"\caption{{{cap}}}" + (rf"\label{{{label}}}" if label else ""),
            r"\begin{tabular}{l" + "c" * 12 + "}", r"\toprule", head1, cm, head2, r"\midrule",
            *body, r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: first row gives the average MSFE of the "
            r"individual forecasts; other rows report ratios to it, the share of units beating the benchmark, "
            r"and the shares with the smallest and largest MSFE. Method of Pesaran, Pick and Timmermann (2026).}",
            r"\end{table}"])
    if which == "dm":
        dm = ev.dm_tests(methods, variance=variance)
        cl = [i for i in dm.index if i.startswith("DM<")][0]
        cu = [i for i in dm.index if i.startswith("DM>")][0]
        cols = " & ".join(_esc(c) for c in dm.columns)
        r1 = "Panel DM & " + " & ".join(f"{v:.2f}" for v in dm.loc["Panel DM"]) + r" \\"
        crit = cu[3:]
        r2 = (rf"$\mathrm{{DM}}<-{crit}$ / $\mathrm{{DM}}>{crit}$ & "
              + " & ".join(f"{int(a)}/{int(b)}" for a, b in zip(dm.loc[cl], dm.loc[cu])) + r" \\")
        return "\n".join([
            r"\begin{table}[htbp]\centering\footnotesize",
            rf"\caption{{{caption or 'Diebold-Mariano tests for equal predictive accuracy'}}}",
            r"\begin{tabular}{l" + "c" * dm.shape[1] + "}", r"\toprule", " & " + cols + r" \\", r"\midrule",
            r1, r2, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    raise ValueError("which must be 'msfe' or 'dm'")


def forecast_table_latex(df: pd.DataFrame, caption: str = "Panel forecasts", float_format="%.3f") -> str:
    return df.to_latex(float_format=float_format, caption=caption, escape=True)


def _esc(s: str) -> str:
    s = str(s).replace("w*_i", "@OMEGA@")
    s = (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
         .replace("_", r"\_").replace("#", r"\#").replace("*", r"$^*$"))
    return s.replace("@OMEGA@", r"$\omega_i^*$")
