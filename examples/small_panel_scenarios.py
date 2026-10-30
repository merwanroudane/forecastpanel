"""Small-N, small-T panel: annual forecasts with scenarios, intervals and evaluation.

The typical applied setting of firm, sector or country panels: a handful of
units observed for about a dozen years, forecasts five years ahead under
alternative scenarios for a common driver. Real data: Q4-on-Q4 real house
price inflation of nine large US metropolitan areas, 2011-2022 (T = 12),
with national house price inflation as a contemporaneous common driver.

Run:  python examples/small_panel_scenarios.py

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

import forecastpanel as fp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# 1. Build an annual panel: N = 9 metros, 2011-2022 -----------------------------
house = fp.load_house_prices()
metros = {35620: "New York", 31080: "Los Angeles", 16980: "Chicago", 19100: "Dallas",
          26420: "Houston", 47900: "Washington", 33100: "Miami", 37980: "Philadelphia",
          12060: "Atlanta"}
d = house[house["msa"].isin(metros) & house["quarter"].str.endswith("Q4")].copy()
d["year"] = d["quarter"].str[:4].astype(int)
d["metro"] = d["msa"].map(metros)
d = d[(d["year"] >= 2010) & (d["year"] <= 2022)][["metro", "year", "dhp", "dhp_us"]]
print(d.pivot(index="year", columns="metro", values="dhp").round(1))

# 2. Model: own lag + contemporaneous national inflation (scenario variable) ----
model = fp.PanelForecaster(d, unit="metro", time="year", y="dhp",
                           y_lags=[1], exog=["dhp_us"],
                           methods=["individual", "pooled", "fe", "eb", "comb_pool",
                                    "comb_unit", "mallows", "stein"])
res = model.fit()
print(res.summary())

# 3. Two scenarios for the national driver, 2023-2027 ----------------------------
years = [2023, 2024, 2025, 2026, 2027]
baseline = pd.DataFrame({"year": years, "dhp_us": [-2.0, 0.0, 1.5, 2.0, 2.0]})
downturn = pd.DataFrame({"year": years, "dhp_us": [-6.0, -4.0, -1.0, 1.0, 2.0]})
fc_base = res.forecast(horizon=5, future=baseline)
fc_down = res.forecast(horizon=5, future=downturn)
table = pd.concat({"Baseline": fc_base["Emp. Bayes"], "Downturn": fc_down["Emp. Bayes"]}, axis=1)
print(table.unstack(0).round(2))
table.unstack(0).round(2).to_csv(os.path.join(OUT, "small_panel_scenarios.csv"))

# 4. Bootstrap predictive intervals (baseline) ------------------------------------
iv = res.intervals(horizon=5, future=baseline, n_boot=499, seed=2026,
                   methods=["eb", "comb_pool", "pooled", "individual"])
print(iv.xs("Emp. Bayes", axis=1, level=0).round(2).head(10))
fig, axes = res.plot_forecasts(methods=["eb", "comb_pool", "pooled", "individual"], history=12,
                               ylabel="% y/y")
fp.save_figure(fig, os.path.join(OUT, "small_panel_forecasts"))

# 5. Pseudo out-of-sample check with an expanding window -------------------------
ev = model.evaluate(window=7, scheme="expanding")
print(ev.summary())
with open(os.path.join(OUT, "small_panel_evaluation.txt"), "w") as fh:
    fh.write(str(ev.summary()))
print("done")
