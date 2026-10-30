"""Monte Carlo experiments of Section 5 (Table 1) with the DGP of Section 5.1.

The paper uses 10,000 replications; ``N_REP`` below keeps the run short.
Pass n_jobs=-1 (requires joblib) to use all cores.

Run:  python examples/monte_carlo_table1.py

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

import forecastpanel as fp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
N_REP = 300

designs = fp.table1_designs(N=100)
designs = designs[(designs.kappa == "zero") & (designs.rho_corr == 0.0)]
rows = []
for _, d in designs.iterrows():
    r = fp.monte_carlo(N=int(d["N"]), T=int(d["T"]), n_rep=N_REP, a_beta=d.a_beta,
                       sigma2_alpha=d.sigma2_alpha, sigma2_gamma=d.sigma2_gamma,
                       rho0=d.rho0, rho_corr=d.rho_corr, kappa=d.kappa, seed=10000)
    rows.append({"a_beta": d.a_beta, "sigma2_alpha": d.sigma2_alpha, "T": int(d["T"]), **r.to_dict()})
    print(rows[-1])
tab = pd.DataFrame(rows)
print(tab.round(3).to_string(index=False))
tab.round(3).to_csv(os.path.join(OUT, "monte_carlo_table1.csv"), index=False)

for (ab, s2a), sub in tab.groupby(["a_beta", "sigma2_alpha"]):
    fig, ax = fp._plots.plot_monte_carlo(sub, methods=["Pooled", "FE", "Emp. Bayes", "Comb. (pool)",
                                                       "Comb. (w*_i)"])
    ax.set_title(f"Monte Carlo, a_beta = {ab}, sigma2_alpha = {s2a}, kappa_i = 0")
    fp.save_figure(fig, os.path.join(OUT, f"mc_abeta{ab}"))
