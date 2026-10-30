# Data sources

All files were produced by `tools/build_datasets.py` from the replication archive of
Pesaran, Pick and Timmermann (2026), *Quantitative Economics* 17(2), doi:10.3982/QE2589
(`panelForecasting_replicationFiles_8Oct2025.zip`), applying exactly the transformations of
`ppt_house.m` and `ppt_cpi.m`.

| File | Content | Original source |
|---|---|---|
| `house_prices.csv.gz` | 377 MSAs x 189 quarters (1976Q1-2023Q1): `dhp` = 100 x log(P_t/P_{t-4}) of the Freddie Mac House Price Index deflated by CPI-U; spatial, regional and national averages | Freddie Mac House Price Index (FMHPI), https://www.freddiemac.com/research/indices/house-price-index ; CPI-U `CPIAUCSL`, FRED, Federal Reserve Bank of St. Louis |
| `house_weights.csv.gz` | Row-normalised spatial weights, MSAs within 100 miles (W100), as an edge list | Yang, C. (2021), Common factors and spatial dependence: an application to US house prices, *Econometric Reviews* 40(1), 14-50 |
| `cpi_subindices.csv.gz` | Monthly inflation 100 x (P_t/P_{t-1} - 1) of 187 CPI sub-indices, 1967M1-2023M5 | US Bureau of Labor Statistics via FRED (public domain) |
| `macro_predictors.csv.gz` | Default spread `dfy` and term spread `tms` (in %), 1947M1-2022M12 | Welch, I. and Goyal, A. (2008), *Review of Financial Studies* 21(4), 1455-1508; data from Amit Goyal's website |

Attribution: the FMHPI is published by Freddie Mac; please cite Freddie Mac when using
the house price series. BLS/FRED CPI series are public-domain US government data.
