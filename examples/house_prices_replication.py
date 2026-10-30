"""Replication of the house-price application of Pesaran, Pick and Timmermann (2026).

Real house price inflation in 362 US metropolitan areas, 1976Q1-2023Q1.
Model (52): y_it = a_i + b_i y_{i,t-1} + b*_i y*_{i,t-1} + g_Ri ybar^R_{i,t-1} + g_Ci ybar^C_{t-1} + e_it,
rolling windows of 60 quarters, 128 one-quarter-ahead forecasts (1991Q2-2023Q1).

Run:  python examples/house_prices_replication.py
Writes tables and figures to docs/assets/.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import os

import matplotlib
matplotlib.use("Agg")

import forecastpanel as fp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# 1. Data --------------------------------------------------------------------
df = fp.load_house_prices(with_neighbors_only=True)
print(df.head())

# 2. Model: lag of own inflation, spatial lag, regional and national averages --
model = fp.PanelForecaster(
    df, unit="msa", time="quarter", y="dhp",
    y_lags=[1], x={"dhp_spatial": 1, "dhp_region": 1, "dhp_us": 1},
    methods="paper",                      # the eight methods of Tables 1-3
)

# 3. Pseudo out-of-sample evaluation exactly as in the paper -------------------
ev = model.evaluate(window=60, n_forecasts=128)
summary = ev.summary()
print(summary)
with open(os.path.join(OUT, "house_table2_3.txt"), "w") as fh:
    fh.write(str(summary))
with open(os.path.join(OUT, "house_table2.tex"), "w") as fh:
    fh.write(ev.to_latex("msfe", caption="House price inflation: out-of-sample accuracy"))

# The published kappa columns: same numbers as Table 2 of the paper
ev_pub = model.evaluate(window=60, n_forecasts=128, kappa_legacy=True)
with open(os.path.join(OUT, "house_table2_published.txt"), "w") as fh:
    fh.write(ev_pub.msfe_table().round(3).to_string())

# 4. Figures -----------------------------------------------------------------
for cond in ("all", "kappa0", "kappa1"):
    fig, ax = ev.plot_density(cond=cond)
    fp.save_figure(fig, os.path.join(OUT, f"house_density_{cond}"))
fig, ax = ev.plot_cumulative()
fp.save_figure(fig, os.path.join(OUT, "house_cumulative"))
fig, ax = ev.plot_heatmap(max_units=45)
fp.save_figure(fig, os.path.join(OUT, "house_heatmap"))

# 5. Forecasts for 2023Q2 - 2024Q1 from the last window ------------------------
res = model.fit(window=60)
print(res.summary())
with open(os.path.join(OUT, "house_fit_summary.txt"), "w") as fh:
    fh.write(str(res.summary()))
fc = res.forecast(horizon=4, fill="last")
big = [31080, 35620, 16980, 19100, 26420, 33100]          # LA, NY, Chicago, Dallas, Houston, Miami
names = df.drop_duplicates("msa").set_index("msa")["msa_name"]
print(fc.loc[big].round(2))
res.intervals(horizon=4, fill="last", n_boot=199, seed=2026, methods=["eb", "comb_pool", "pooled"])
fig, axes = res.plot_forecasts(units=big, methods=["eb", "comb_pool", "pooled", "individual"],
                               history=24, ylabel="% y/y")
for ax, u in zip(axes.ravel(), big):
    ax.set_title(names[u], fontsize=9)
fp.save_figure(fig, os.path.join(OUT, "house_forecasts"))
fig, ax = res.plot_coefficients("dhp(-1)")
fp.save_figure(fig, os.path.join(OUT, "house_coefficients"))
fig, ax = res.plot_weights()
fp.save_figure(fig, os.path.join(OUT, "house_weights"))
print("done")
