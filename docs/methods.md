# Methods: step -> equation map

Package `forecastpanel` implements Pesaran, M. H., Pick, A. and Timmermann, A. (2026),
"Forecasting with panel data: Estimation uncertainty versus parameter heterogeneity",
*Quantitative Economics* 17(2), 342-393, doi:10.3982/QE2589, and reproduces the authors'
MATLAB replication code (Pick, 2025).

Notation: `Y (N,T)`, `W (N,T,K)` with `w_it = (1, x_it')'`, `w_next = w_{i,T+1}`.

## Forecasts

| Paper | Function | Notes |
|---|---|---|
| (11), (S.10)-(S.11) individual OLS | `individual_ols` | `pinv(W_i'W_i) W_i'y_i` as in the code; s_i^2 with T-K |
| (12), (S.13)-(S.14) pooled OLS | `pooled_ols` | |
| (25), (S.15) fixed effects | `fixed_effects` | within estimator, alpha_i = mean(y_i - X_i b) |
| (S.16) random effects + Goldberger BLUP | `random_effects_blup` | FE-based variance components; negative s2_alpha replaced by var(alpha_FE) (code) |
| (44)-(45) empirical Bayes | `empirical_bayes` | Omega with 1/N; shrinkage matrix W_iT returned |
| (32), (40), (41), (43) common weight ind/pooled | `weight_individual_pooled` | half-jackknife (Sec. 4.3); odd T drops the middle observation |
| (39), (79)-(81) common weight ind/FE | `weight_individual_fe` | `variant='code'` (default) or `'paper'` |
| (36), (46) unit weights | `weights_unit_specific` | sigma_i^2 of PPT (2022), Omega with ddof=1 (code) |
| Table S.4 equal weights | `panel_forecasts(..., ['eq_pool','eq_fe'])` | |
| Table S.4 oracle weights | `oracle_weights` | true parameters and disturbances |
| (S.17) hierarchical Bayes | `hierarchical_bayes` | Gibbs sampler of the code; 1500 draws, 500 burn-in |
| Sec. 5.1 DGP (47)-(51) | `draw_parameters`, `simulate_panel`, `monte_carlo` | parameters drawn once, data redrawn |
| Table S.1 PR^2 | `panel_pr2` | |
| Appendix D, Table A.1 | `delta_ar_numerical` | |
| Sec. 6.1 evaluation, kappa_i | `evaluate_forecasts`, `EvaluationResults.msfe_table` | |
| Table 3 panel DM | `panel_dm_test` | Pesaran, Pick and Pranovich (2013, App. 2) |
| Table 3 unit DM | `dm_test` | Diebold-Mariano (1995); HLN (1997) also returned |
| Figure 1 | `EvaluationResults.plot_density` | normal kernel, bandwidth 0.03 (code) |

## Combination weight (eq. 32)

omega = (Delta - psi/T) / (Delta + h/T - 2 psi/T), truncated to [0, 1], with

* Delta = N^-1 sum_i (w_{i,T+1}'(theta_tilde - theta_hat_i))^2  (40)
* h = N^-1 sum_i s_i^2 w_{i,T+1}' Q_iT^-1 w_{i,T+1}, Q_iT = W_i'W_i/T  (41)
* psi = [T N^-1 sum_i b_i' w w'] Qbar^-1 qbar - T N^-1 sum_i b_i' w w' eta_i, with
  b_i = (theta_ia + theta_ib)/2 - theta_hat_i (half-jackknife bias), eta_i = theta_hat_i - mean(theta_hat), qbar = (NT)^-1 sum W_i'W_i eta_i  (43)

## Where the code and the paper differ (and what the package does)

1. **kappa_i classification, house prices (Table 2, upper panel).** `ppt_house.m` stores
   d_{i,T+1} with `wthetaiTp1{1,ctr}(:,i) = ...`, filling a whole column; the tabulation reads
   `wthetaiTp1{1,tt}(ii)`, i.e. unit 1's value for every unit. All "kappa_i = 0" and
   "kappa_i = +/-1" columns of the house-price panel are computed against unit 1's
   d_{1,T+1}. The package classifies each unit with its own d_{i,T+1} (as Sec. 6.1 describes);
   `kappa_legacy=True` reproduces the published numbers exactly. The "all" columns and Table 3
   are unaffected. The CPI code stores the value correctly.
2. **Panel DM for unbalanced panels (CPI, Table 3).** In `PanelDieboldMarianoPPP_matlab.m` the
   unbalanced branch scales the variance by sum_j 1/T_j (= N/T) instead of 1/T_i, which divides
   the statistic by about sqrt(N). The balanced branch (house prices) is correct. The package
   uses the correct formula; `variance='ppp-legacy'` reproduces the code. Footnote 17 describes
   a third variant (HAC on the cross-sectional average loss differential), available as
   `variance='cross-section-average'`.
3. **Principal-component factor in the CPI application.** In `ppt_cpi.m` the factor is
   computed from `PriceData(:, 1:maxlag+rollwin)` for every forecast origin, so the same
   1967-1973 factor values enter every rolling window. Users who add a factor should
   re-estimate it in each window (`principal_components`).
4. **FE combination weight.** The code uses eta_i = beta_i - beta_FE without the
   qbar'Qbar^-1 qbar term, the individual half-jackknife, and drops the (c_FE - c_beta)
   term (O(T^-1)); eqs. (79)-(81) use mean-group eta with the q-term and the FE half-jackknife
   of Chudik, Pesaran and Yang (2018). Default: code; `fe_weight='paper'`: equations.
   With OLS residuals that include an intercept, the c-difference term is identically zero.
5. **x_{i,T+1} in the Monte Carlo.** The code shifts x by kappa_i sigma_xi^2 (variance), the text
   by sigma_xi. Default: code; `x_shift='sd'` follows the text.
6. **Figure 1 bandwidth.** The figure note says 0.04, the code uses 0.03 (default 0.03).
7. **Unit weights.** The Supplement states Omega with 1/N; the code uses `cov` (1/(N-1)).
   Default follows the code (`ddof=1`).

## Extensions beyond the paper (all optional)

| Method key | Reference |
|---|---|
| `comb_unit_ba` bias-adjusted unit weights | Pesaran, Pick and Timmermann (2022, CEPR DP) - in the code |
| `mean_group` | Pesaran and Smith (1995) |
| `ib` iterative empirical Bayes | Maddala, Trost, Li and Joutz (1997); Hsiao, Pesaran and Tahmiscioglu (1999) |
| `stein` Stein-rule shrinkage | Maddala et al. (1997); Judge and Bock (1978) |
| `mallows` Mallows pooling averaging (two candidates, closed form) | Wang, Zhang and Paap (2019) |
| `re_ar1` RE BLUP with AR(1) remainder | Baltagi and Li (1992); Baltagi (2008, eq. 24) |
| `omega_regularize` ridge/diagonal Omega | for N close to K (paper footnote 12) |
| iterated multi-step forecasts with scenario paths | - |
| bootstrap predictive intervals (fixed or recursive design) | - |
| expanding-window evaluation, multi-step evaluation | - |

The Mallows weight on the pooled estimator solves min_w w^2 D + 2[w sum s_i^2 tr(S^-1 A_i) +
(1-w) K sum s_i^2], A_i = W_i'W_i, S = sum A_i, D = sum (theta_pool - theta_i)'A_i(theta_pool - theta_i):
w* = sum_i s_i^2 [K - tr(S^-1 A_i)] / D.
