"""Build the packaged datasets from the Pesaran-Pick-Timmermann replication archive.

Usage:  python tools/build_datasets.py <path/to/panelForecasting>

Reproduces the transformations of ``ppt_house.m`` and ``ppt_cpi.m`` exactly:
house prices are deflated by the CPI and transformed to 4-quarter log changes
(x100); CPI sub-indices are transformed to monthly percentage changes.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

src = sys.argv[1]
out = os.path.join(os.path.dirname(__file__), "..", "src", "forecastpanel", "datasets")
H = os.path.join(src, "data", "house")
C = os.path.join(src, "data", "cpi")

# ---------------------------------------------------------------- house prices
m = sio.loadmat(os.path.join(H, "HousePriceDataJun2023.mat"))
cpi = np.real(sio.loadmat(os.path.join(H, "US_CPI_Jun2023.mat"))["cpi"][:, 1])
hp, msa = m["dhp"], m["dmsa"][0].astype(int)
N = hp.shape[1]
raw = pd.read_csv(os.path.join(H, "data_main_no_header.csv"), header=None).iloc[1:, 1:].to_numpy(float)
region = raw[:N, 1].astype(int)                                   # csvread(..,0,1); data(:,2)
W100 = pd.read_csv(os.path.join(H, "W100.csv"), header=None).to_numpy(float)
rhp = hp / cpi[:, None]
dhp = np.log(rhp[4:] / rhp[:-4]) * 100                            # 189 x 377
reg_avg = np.column_stack([dhp[:, region == region[i]].mean(axis=1) for i in range(N)])
us_avg = dhp.mean(axis=1)
spatial = (W100 @ dhp.T).T
nn = (W100 > 0).sum(axis=1)
quarters = pd.period_range("1976Q1", periods=dhp.shape[0], freq="Q").astype(str)
f = pd.read_csv(os.path.join(H, "fmhpi_master_file.csv"), usecols=["GEO_Type", "GEO_Name", "GEO_Code"])
names = (f[f.GEO_Type == "CBSA"].drop_duplicates("GEO_Code").assign(GEO_Code=lambda d: d.GEO_Code.astype(int))
         .set_index("GEO_Code")["GEO_Name"])
rows = []
for i in range(N):
    rows.append(pd.DataFrame({
        "msa": msa[i], "msa_name": names.get(msa[i], ""), "region": region[i],
        "n_neighbors": nn[i], "quarter": quarters, "dhp": dhp[:, i], "dhp_spatial": spatial[:, i],
        "dhp_region": reg_avg[:, i], "dhp_us": us_avg}))
house = pd.concat(rows, ignore_index=True)
house.to_csv(os.path.join(out, "house_prices.csv.gz"), index=False, float_format="%.10g")
i_, j_ = np.nonzero(W100)
pd.DataFrame({"msa": msa[i_], "neighbor": msa[j_], "weight": W100[i_, j_]}).to_csv(
    os.path.join(out, "house_weights.csv.gz"), index=False, float_format="%.10g")
print("house", house.shape)

# ---------------------------------------------------------------- CPI
cp = pd.read_csv(os.path.join(C, "CPIsubindices.csv"))
months = pd.period_range("1947-01", periods=len(cp), freq="M")
cp.index = months
lv = cp.drop(columns="Dates").apply(pd.to_numeric, errors="coerce")
infl = 100 * (lv / lv.shift(1) - 1)
cats = {}
with open(os.path.join(C, "BLS_categories.txt"), encoding="utf-8-sig") as fh:
    for line in fh:
        if ";" in line:
            a, b = line.rstrip("\n").split(";", 1)
            cats[a.strip()] = b.strip()
long = infl.stack(future_stack=True).rename("infl").reset_index()
long.columns = ["month", "series", "infl"]
long["month"] = long["month"].astype(str)
long = long[long["month"] >= "1967-01"].dropna(subset=["infl"])
long.to_csv(os.path.join(out, "cpi_subindices.csv.gz"), index=False, float_format="%.10g")
mp = pd.read_csv(os.path.join(C, "MacroPredictors.csv"), encoding="utf-8-sig").dropna(subset=["yyyymm"])
mp = pd.DataFrame({"month": pd.PeriodIndex([f"{int(v)//100}-{int(v)%100:02d}" for v in mp.yyyymm], freq="M").astype(str),
                   "dfy": mp["default-spread"] * 100, "tms": mp["term-spread"] * 100})
mp.to_csv(os.path.join(out, "macro_predictors.csv.gz"), index=False, float_format="%.10g")
print("cpi", long.shape, long.series.nunique(), "macro", mp.shape)
