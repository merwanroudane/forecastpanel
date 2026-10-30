# Changelog

## [0.1.1] - 2026-10-30

### Fixed
- `ForecastResults.summary()` no longer requires the optional `jinja2` package: the
  coefficient table is rendered in LaTeX by the package's own booktabs builder.

### Changed
- Minimum Python version is 3.10 (tested on 3.10-3.13, Linux and Windows).

## [0.1.0] - 2026-10-30

### Added
- Individual, pooled, fixed-effects, random-effects (Goldberger BLUP) and empirical Bayes
  panel forecasts, and the combination forecasts of Pesaran, Pick and Timmermann (2026):
  common individual/pooled weight (eq. 32), individual/FE weight (eq. 39), unit-specific
  weights (eq. 36/46) and equal weights.
- Extensions: bias-adjusted unit weights, mean group, iterative Bayes and Stein-rule shrinkage
  (Maddala et al., 1997), Mallows pooling averaging (Wang, Zhang and Paap, 2019), RE BLUP with
  AR(1) errors (Baltagi and Li, 1992), hierarchical Bayes (Gibbs sampler), ridge/diagonal
  regularisation of Omega for small N.
- `PanelData` with automatic lags, contemporaneous scenario variables and trends; iterated
  multi-step forecasts, scenario paths, additive offsets and bootstrap predictive intervals.
- Rolling/expanding pseudo out-of-sample evaluation reproducing Table 2, Table 3 and Figure 1;
  panel Diebold-Mariano test (Pesaran, Pick and Pranovich, 2013) and unit DM/HLN tests.
- Monte Carlo design of Section 5.1, pooled R-squared (Table S.1), oracle weights (Table S.4).
- House-price and CPI datasets of the paper's applications.
- Journal-style text/LaTeX tables and figures with the MATLAB Parula colour map.

### Verified
- Forecasts agree with the authors' MATLAB replication code to 1e-10 and the published Tables 2
  and 3 (house prices) are reproduced exactly.
