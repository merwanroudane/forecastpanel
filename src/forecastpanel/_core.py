"""Numerical core: panel estimators and forecast-combination weights.

Every function maps to a numbered equation of

    Pesaran, M. H., Pick, A. and Timmermann, A. (2026). Forecasting with panel
    data: Estimation uncertainty versus parameter heterogeneity.
    Quantitative Economics 17(2), 342-393. doi:10.3982/QE2589

or of its Supplemental Appendix (eqs. S.10-S.17), and reproduces the authors'
MATLAB replication code (Pick, 2025) step by step. Where the paper and the
code differ, the code is the default (it produced the published tables) and
the paper's literal formula is offered as an option; each case is flagged
with ``# PAPER vs CODE``.

Array conventions (0-based, balanced panel)
-------------------------------------------
Y      : (N, T)     outcomes y_it, t = 1..T of the estimation window
W      : (N, T, K)  regressors w_it = (1, x_it')', intercept in column 0
w_next : (N, K)     regressors of the forecast period, w_{i,T+1}

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

__all__ = [
    "individual_ols", "pooled_ols", "fixed_effects", "random_effects_blup",
    "random_effects_ar1_blup", "empirical_bayes", "iterative_bayes",
    "mean_group", "hierarchical_bayes", "half_jackknife",
    "weight_individual_pooled", "weight_individual_fe", "weights_unit_specific",
    "weight_mallows", "weight_stein", "oracle_weights",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clip01(w: float) -> float:
    # The code truncates every estimated weight to [0, 1].
    return float(min(max(w, 0.0), 1.0))


def _mldivide(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares solve, the analogue of MATLAB's ``A \\ b``."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


@dataclass
class IndividualFit:
    """Unit-by-unit OLS, eq. (11) and (S.11)."""

    theta: np.ndarray        # (N, K)  theta_hat_i
    sigma2: np.ndarray       # (N,)    sigma_hat_i^2 = e_i'e_i / (T - K)
    resid: np.ndarray        # (N, T)
    WW: np.ndarray           # (N, K, K)  W_i'W_i
    Wy: np.ndarray           # (N, K)     W_i'y_i
    rank_deficient: np.ndarray  # (N,) bool


def individual_ols(Y: np.ndarray, W: np.ndarray) -> IndividualFit:
    """Individual least squares, theta_hat_i = (W_i'W_i)^{-1} W_i'y_i (eq. 11).

    Uses the Moore-Penrose inverse exactly as the replication code
    (``pinv(zi'*zi)*(zi'*yi)``), so a rank-deficient unit still returns the
    minimum-norm solution; such units are flagged in ``rank_deficient``.
    """
    N, T, K = W.shape
    WW = np.einsum("itk,itl->ikl", W, W)
    Wy = np.einsum("itk,it->ik", W, Y)
    theta = np.empty((N, K))
    rd = np.zeros(N, dtype=bool)
    for i in range(N):
        theta[i] = np.linalg.pinv(WW[i]) @ Wy[i]
        rd[i] = np.linalg.matrix_rank(WW[i]) < K
    resid = Y - np.einsum("itk,ik->it", W, theta)
    sigma2 = (resid ** 2).sum(axis=1) / (T - K)
    return IndividualFit(theta, sigma2, resid, WW, Wy, rd)


def pooled_ols(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pooled least squares, theta_tilde = (W'W)^{-1} W'y (eq. 12, S.14)."""
    WW = np.einsum("itk,itl->kl", W, W)
    Wy = np.einsum("itk,it->k", W, Y)
    return np.linalg.pinv(WW) @ Wy


@dataclass
class FEFit:
    """Fixed-effects (within) estimates, eq. (25)."""

    alpha: np.ndarray   # (N,)  alpha_hat_{i,FE}
    beta: np.ndarray    # (k,)  beta_hat_FE (slopes)

    @property
    def theta(self) -> np.ndarray:
        return np.column_stack([self.alpha, np.tile(self.beta, (len(self.alpha), 1))])


def fixed_effects(Y: np.ndarray, W: np.ndarray) -> FEFit:
    """Within estimator and unit intercepts (eq. 25, S.15).

    beta_FE = (sum_i X_i'M_T X_i)^{-1} sum_i X_i'M_T y_i and
    alpha_i = tau_T'(y_i - X_i beta_FE)/T  (code: ``FEforecastLargeN.m``).
    """
    X = W[:, :, 1:]
    Xd = X - X.mean(axis=1, keepdims=True)
    yd = Y - Y.mean(axis=1, keepdims=True)
    N, T, k = X.shape
    beta = _mldivide(Xd.reshape(N * T, k), yd.reshape(N * T))
    alpha = (Y - np.einsum("itk,k->it", X, beta)).mean(axis=1)
    return FEFit(alpha, beta)


@dataclass
class REFit:
    """Goldberger (1962) random-effects BLUP, eq. (S.16)."""

    theta_gls: np.ndarray    # (K,)  (alpha_RE, beta_RE')'
    blup_shift: np.ndarray   # (N,)  T s2_a/(T s2_a + s2_e) * mean_t u_it
    sigma2_eps: float
    sigma2_alpha: float

    @property
    def theta(self) -> np.ndarray:
        th = np.tile(self.theta_gls, (len(self.blup_shift), 1))
        th[:, 0] += self.blup_shift
        return th


def random_effects_blup(Y: np.ndarray, W: np.ndarray) -> REFit:
    """Random-effects GLS with Goldberger's BLUP correction (eq. S.16).

    Follows ``GoldbergerREBLUP_FEinit.m``: variance components from the FE
    residuals; if the between-variance estimate is negative it is replaced by
    the sample variance of the FE intercepts (code fallback, not in the paper).
    The forecast is w'theta_RE + [s2_a/(T s2_a + s2_e)] sum_t u_it.
    """
    N, T, K = W.shape
    k = K - 1
    X = W[:, :, 1:]
    fe = fixed_effects(Y, W)
    ymxb = Y - np.einsum("itk,k->it", X, fe.beta)
    sum1 = ((ymxb - ymxb.mean(axis=1, keepdims=True)) ** 2).sum()
    sum2 = ((Y.mean(axis=1) - X.mean(axis=1) @ fe.beta) ** 2).sum()
    s2e = sum1 / (N * (T - 1) - k)
    s2a = sum2 / (N - k) - s2e / T
    if s2a < 0:
        s2a = float(np.var(fe.alpha, ddof=1))
    psi = s2e / (T * s2a + s2e)
    # Omega^{-1} = s2e^{-1} (M_T + psi P_T): apply via within/between parts.
    Wm = W.mean(axis=1, keepdims=True)
    ym = Y.mean(axis=1, keepdims=True)
    Wd, yd = W - Wm, Y - ym
    XX = (np.einsum("itk,itl->kl", Wd, Wd) + psi * T * np.einsum("ik,il->kl", Wm[:, 0], Wm[:, 0])) / s2e
    Xy = (np.einsum("itk,it->k", Wd, yd) + psi * T * np.einsum("ik,i->k", Wm[:, 0], ym[:, 0])) / s2e
    theta = np.linalg.solve(XX, Xy)
    u = Y - np.einsum("itk,k->it", W, theta)
    shift = s2a / (T * s2a + s2e) * u.sum(axis=1)
    return REFit(theta, shift, float(s2e), float(s2a))


@dataclass
class REAR1Fit:
    """Random effects with AR(1) remainder disturbances (Baltagi and Li, 1992)."""

    theta_gls: np.ndarray
    rho: float
    sigma2_mu: float
    sigma2_eps: float
    u_last: np.ndarray       # (N,) GLS residual at T
    mu_shift: np.ndarray     # (N,) individual-effect part of the BLUP correction

    def correction(self, h: int = 1) -> np.ndarray:
        """BLUP correction for horizon h: rho^h u_iT + mu-part (Baltagi 2008, eq. 24)."""
        return self.rho ** h * self.u_last + self.mu_shift


def random_effects_ar1_blup(Y: np.ndarray, W: np.ndarray) -> REAR1Fit:
    """Error-components model with AR(1) remainder and its BLUP.

    EXTENSION (not in Pesaran-Pick-Timmermann). Model u_it = mu_i + v_it,
    v_it = rho v_{i,t-1} + e_it. rho is estimated from the within residuals;
    the Prais-Winsten transformed model is estimated by feasible GLS
    (Baltagi and Li, 1991) and the one-step BLUP correction is
    rho u_iT + [(1-rho)^2 s2_mu / s2_a] (omega u*_i1 + sum_{t>=2} u*_it),
    omega = sqrt((1+rho)/(1-rho)), s2_a = d^2 (1-rho)^2 s2_mu + s2_e,
    d^2 = omega^2 + T - 1 (Baltagi, 2008, eq. 24). For h > 1 the AR part
    decays as rho^h.
    """
    N, T, K = W.shape
    fe = fixed_effects(Y, W)
    X = W[:, :, 1:]
    e = Y - np.einsum("itk,k->it", X, fe.beta)
    e = e - e.mean(axis=1, keepdims=True)
    rho = float((e[:, 1:] * e[:, :-1]).sum() / (e[:, :-1] ** 2).sum())
    rho = min(max(rho, -0.99), 0.99)
    c1 = np.sqrt(1 - rho ** 2)

    def pw(a):
        out = a.copy()
        out[:, 0] = c1 * a[:, 0]
        out[:, 1:] = a[:, 1:] - rho * a[:, :-1]
        return out

    Ys = pw(Y)
    Ws = np.stack([pw(W[:, :, j]) for j in range(K)], axis=2)
    omega = np.sqrt((1 + rho) / (1 - rho))
    iota = np.r_[omega, np.ones(T - 1)]
    d2 = float(iota @ iota)
    P = np.outer(iota, iota) / d2
    # QUE variance components from transformed OLS residuals
    th_ols = np.linalg.pinv(np.einsum("itk,itl->kl", Ws, Ws)) @ np.einsum("itk,it->k", Ws, Ys)
    us = Ys - np.einsum("itk,k->it", Ws, th_ols)
    s2a = float(np.einsum("it,ts,is->", us, P, us) / N)
    s2e = float(np.einsum("it,ts,is->", us, np.eye(T) - P, us) / (N * (T - 1)))
    s2a = max(s2a, s2e)
    theta_fb = 1.0 - np.sqrt(s2e / s2a)
    # Fuller-Battese transform on the PW data
    Yss = Ys - theta_fb * (Ys @ P)
    Wss = Ws - theta_fb * np.einsum("ts,isk->itk", P, Ws)
    theta = np.linalg.pinv(np.einsum("itk,itl->kl", Wss, Wss)) @ np.einsum("itk,it->k", Wss, Yss)
    s2mu = max((s2a - s2e) / (d2 * (1 - rho) ** 2), 0.0)
    u = Y - np.einsum("itk,k->it", W, theta)
    ustar = pw(u)
    mu_part = (1 - rho) ** 2 * s2mu / s2a * (omega * ustar[:, 0] + ustar[:, 1:].sum(axis=1))
    return REAR1Fit(theta, rho, s2mu, s2e, u[:, -1], mu_part)


def mean_group(theta_i: np.ndarray) -> np.ndarray:
    """Mean-group estimator, mean of theta_hat_i (Pesaran and Smith, 1995)."""
    return theta_i.mean(axis=0)


def _omega_eta(theta_i: np.ndarray, ddof: int = 0, regularize: Optional[str] = None,
               ridge: float = 1e-6) -> np.ndarray:
    dev = theta_i - theta_i.mean(axis=0)
    om = dev.T @ dev / (theta_i.shape[0] - ddof)
    if regularize == "diagonal":
        om = np.diag(np.diag(om))
    elif regularize == "ridge":
        om = om + ridge * np.trace(om) / om.shape[0] * np.eye(om.shape[0])
    return om


@dataclass
class EBFit:
    """Empirical Bayes estimates, eqs. (44)-(45)."""

    theta: np.ndarray         # (N, K) theta_hat_{i,EB}
    theta_bar: np.ndarray     # (K,)   mean-group centre
    omega: np.ndarray         # (K, K) Omega_hat_eta
    shrinkage: np.ndarray     # (N, K, K) W_iT, weight on theta_hat_i (eq. 45)


def empirical_bayes(Y: np.ndarray, W: np.ndarray, ind: Optional[IndividualFit] = None,
                    omega_regularize: Optional[str] = None) -> EBFit:
    """Empirical Bayes estimator of Hsiao, Pesaran and Tahmiscioglu (1999), eq. (44).

    theta_EB,i = (s_i^-2 W_i'W_i + Om^-1)^{-1}(s_i^-2 W_i'y_i + Om^-1 theta_bar),
    Om = N^-1 sum (theta_i - theta_bar)(theta_i - theta_bar)'  (``panelEmpiricalBayesForecasts.m``).

    ``omega_regularize`` ('ridge' or 'diagonal') is an EXTENSION for panels with
    N close to K, where Omega_hat is singular (paper footnote 12).
    """
    ind = individual_ols(Y, W) if ind is None else ind
    N, T, K = W.shape
    tbar = ind.theta.mean(axis=0)
    om = _omega_eta(ind.theta, ddof=0, regularize=omega_regularize)
    try:
        iom = np.linalg.inv(om)
        if not np.all(np.isfinite(iom)) or np.linalg.cond(om) > 1e14:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Omega_eta (dispersion of the individual estimates) is singular: N="
            f"{N} units for K={K} coefficients. Use fewer regressors or "
            "omega_regularize='ridge' / 'diagonal'.") from exc
    theta = np.empty((N, K))
    shrink = np.empty((N, K, K))
    for i in range(N):
        A = ind.WW[i] / ind.sigma2[i] + iom
        Ainv = np.linalg.inv(A)
        theta[i] = Ainv @ (ind.Wy[i] / ind.sigma2[i] + iom @ tbar)
        shrink[i] = Ainv @ (ind.WW[i] / ind.sigma2[i])     # eq. (45)
    return EBFit(theta, tbar, om, shrink)


def iterative_bayes(Y: np.ndarray, W: np.ndarray, max_iter: int = 200, tol: float = 1e-8,
                    omega_regularize: Optional[str] = None) -> EBFit:
    """Iterative empirical Bayes (Maddala, Trost, Li and Joutz, 1997; Hsiao et al., 1999).

    EXTENSION. Starts from eq. (44) and iterates: theta_bar = mean(theta_EB),
    Omega = N^-1 sum (theta_EB,i - theta_bar)(.)' , sigma_i^2 from the EB
    residuals, until the coefficients converge.
    """
    ind = individual_ols(Y, W)
    eb = empirical_bayes(Y, W, ind, omega_regularize)
    N, T, K = W.shape
    theta = eb.theta
    for _ in range(max_iter):
        tbar = theta.mean(axis=0)
        om = _omega_eta(theta, 0, omega_regularize) + 1e-12 * np.eye(K)
        iom = np.linalg.inv(om)
        res = Y - np.einsum("itk,ik->it", W, theta)
        s2 = (res ** 2).sum(axis=1) / (T - K)
        new = np.empty_like(theta)
        for i in range(N):
            new[i] = np.linalg.solve(ind.WW[i] / s2[i] + iom, ind.Wy[i] / s2[i] + iom @ tbar)
        done = np.max(np.abs(new - theta)) < tol
        theta = new
        if done:
            break
    shrink = np.stack([np.linalg.solve(ind.WW[i] / s2[i] + iom, ind.WW[i] / s2[i]) for i in range(N)])
    return EBFit(theta, theta.mean(axis=0), om, shrink)


def hierarchical_bayes(Y: np.ndarray, W: np.ndarray, w_next: np.ndarray, *,
                       n_draws: int = 1500, burn_in: int = 500,
                       prior_mean_precision: float = 1e-6, prior_omega_scale: float = 10.0,
                       prior_s2: float = 0.1, prior_nu_s: float = 0.1,
                       rng: Optional[np.random.Generator] = None):
    """Hierarchical Bayes forecasts of Lindley and Smith (1972), Gibbs sampler (S.17).

    Mirrors ``panelHierarchBayesForecasts.m``: theta_i ~ N(theta_bar, Om),
    theta_bar ~ N(0, S_bar) with precision ``prior_mean_precision`` * I,
    Om^{-1} ~ Wishart(K, (K S_Om)^{-1}) with S_Om = ``prior_omega_scale`` * I,
    sigma^2 ~ invGamma(nu/2, nu s^2/2). Starting values as in the code.
    The three priors of the Supplement are (1e-6, 10), (1e-2, 100), (1, 1).

    Returns
    -------
    yhat : (N,) posterior-mean forecasts, theta_post : (N, K) posterior means.
    """
    from scipy.stats import wishart

    rng = np.random.default_rng() if rng is None else rng
    N, T, K = W.shape
    ind = individual_ols(Y, W)
    betas = ind.theta.copy()
    mub = betas.mean(axis=0)
    dev = betas - mub
    isig = np.linalg.inv(dev.T @ dev / N)
    sig2 = float((ind.resid ** 2).sum() / (N * T - K))
    prior_iS = prior_mean_precision * np.eye(K)
    prior_V = prior_omega_scale * np.eye(K)
    nu_b = K
    yhat = np.zeros(N)
    theta_post = np.zeros((N, K))
    keep = n_draws - burn_in
    for rep in range(2, n_draws + 1):
        ssr = 0.0
        for i in range(N):
            Di = np.linalg.inv(ind.WW[i] / sig2 + isig)
            Di = (Di + Di.T) / 2
            b = Di @ (ind.Wy[i] / sig2 + isig @ mub) + np.linalg.cholesky(Di) @ rng.standard_normal(K)
            betas[i] = b
            ssr += float(((Y[i] - W[i] @ b) ** 2).sum())
        if rep > burn_in:
            yhat += (w_next * betas).sum(axis=1) / keep
            theta_post += betas / keep
        df = (T * N + prior_nu_s) / 2
        s = 0.5 * (ssr + prior_nu_s * prior_s2)
        sig2 = 1.0 / rng.gamma(df, 1.0 / s)
        bbar = betas.mean(axis=0)
        V = np.linalg.inv(N * isig + prior_iS)
        V = (V + V.T) / 2
        mub = V @ (N * isig @ bbar + prior_iS @ np.zeros(K)) + np.linalg.cholesky(V) @ rng.standard_normal(K)
        D = betas - mub
        Wm = D.T @ D + prior_V * nu_b
        scale = np.linalg.inv(Wm)
        isig = np.atleast_2d(wishart(df=N + nu_b, scale=(scale + scale.T) / 2).rvs(random_state=rng))
    return yhat, theta_post


# ---------------------------------------------------------------------------
# Half-panel jackknife (Dhaene and Jochmans, 2015; Chudik, Pesaran and Yang, 2018)
# ---------------------------------------------------------------------------
def _halves(T: int):
    """Index sets of the two half samples; the middle observation is dropped
    when T is odd (code: ``Thalf=(T-1)/2; Tsecondhalf=Thalf+2``)."""
    th = T // 2
    a = np.arange(0, th)
    b = np.arange(th + (T % 2), T)
    return a, b


def half_jackknife(Y: np.ndarray, W: np.ndarray, theta_i: np.ndarray) -> np.ndarray:
    """Half-jackknife estimates theta_JK,i = 2 theta_i - (theta_ia + theta_ib)/2 (Sec. 4.3)."""
    N, T, K = W.shape
    a, b = _halves(T)
    out = np.empty_like(theta_i)
    for i in range(N):
        ta = _mldivide(W[i, a], Y[i, a])
        tb = _mldivide(W[i, b], Y[i, b])
        out[i] = 2 * theta_i[i] - 0.5 * (ta + tb)
    return out


def _fe_half_jackknife(Y: np.ndarray, W: np.ndarray, beta_fe: np.ndarray) -> np.ndarray:
    a, b = _halves(Y.shape[1])
    ba = fixed_effects(Y[:, a], W[:, a]).beta
    bb = fixed_effects(Y[:, b], W[:, b]).beta
    return 2 * beta_fe - 0.5 * (ba + bb)


# ---------------------------------------------------------------------------
# Combination weights
# ---------------------------------------------------------------------------
@dataclass
class WeightResult:
    """Estimated combination weight with the components of its formula."""

    weight: float                  # clipped to [0, 1]
    raw: float                     # before clipping
    components: Dict[str, float] = field(default_factory=dict)


def weight_individual_pooled(Y: np.ndarray, W: np.ndarray, w_next: np.ndarray,
                             ind: IndividualFit, theta_pool: np.ndarray) -> WeightResult:
    """Common weight on the individual forecast, eq. (32), estimated by (40), (41), (43).

    omega = (Delta - psi/T) / (Delta + h/T - 2 psi/T), clipped to [0, 1], with
      Delta = N^-1 sum w'eta_i eta_i'w,  eta_i = theta_pool - theta_i      (40)
      h     = N^-1 sum w'Q_iT^-1 H_iT Q_iT^-1 w, H_iT = s_i^2 Q_iT        (41)
      psi   = [T N^-1 sum b_i' w w'] Qbar^-1 qbar(eta_MG)
              - T N^-1 sum b_i' w w' eta_MG,i,   b_i = (theta_ia+theta_ib)/2 - theta_i  (43)
    (``optimalWeights_individual_pooled.m``).
    """
    N, T, K = W.shape
    jk = half_jackknife(Y, W, ind.theta)
    bias = ind.theta - jk                          # = (theta_a+theta_b)/2 - theta_i
    eta_mg = ind.theta - ind.theta.mean(axis=0)
    eta_p = ind.theta - theta_pool
    Q = ind.WW.sum(axis=0) / (N * T)                                 # Qbar_NT (8)
    q = np.einsum("ikl,il->k", ind.WW, eta_mg) / (N * T)             # qbar_NT (9)
    wb = (bias * w_next).sum(axis=1)                                  # b_i'w
    psi1 = T / N * (wb[:, None] * w_next).sum(axis=0)                 # row vector
    psi2 = T / N * (wb * (w_next * eta_mg).sum(axis=1)).sum()
    psi = float(psi1 @ np.linalg.solve(Q, q) - psi2)
    QiT_inv_w = np.stack([np.linalg.solve(ind.WW[i] / T, w_next[i]) for i in range(N)])
    # w'Q^-1 (s2 Q) Q^-1 w = s2 w'Q^-1 w
    h = float((ind.sigma2 * (QiT_inv_w * w_next).sum(axis=1)).mean())
    delta = float(((eta_p * w_next).sum(axis=1) ** 2).mean())
    raw = (delta - psi / T) / (delta + h / T - 2 * psi / T)
    return WeightResult(_clip01(raw), float(raw), {"Delta": delta, "h": h, "psi": psi, "T": T})


def weight_individual_fe(Y: np.ndarray, W: np.ndarray, w_next: np.ndarray,
                         ind: IndividualFit, fe: FEFit, variant: str = "code") -> WeightResult:
    """Common weight on the individual forecast vs the FE forecast, eq. (39).

    ``variant='code'`` (default) reproduces ``optimalWeights_individual_FE.m``:
    Delta_FE uses eta_i = beta_i - beta_FE without the q'Q^-1 q term, psi_FE uses
    the individual half-jackknife and the (c_FE - c_beta) = O(T^-1) term is
    dropped. ``variant='paper'`` uses eqs. (79)-(81) literally (mean-group eta,
    subtraction of qbar'Qbar^-1 qbar, FE half-jackknife of Chudik et al. 2018,
    and the c-difference term).  # PAPER vs CODE
    """
    N, T, K = W.shape
    k = K - 1
    X = W[:, :, 1:]
    xf = w_next[:, 1:]
    Xd = X - X.mean(axis=1, keepdims=True)
    xfd = xf - X.mean(axis=1)                                         # xbarbar_{i,T+1}
    XMX = np.einsum("itk,itl->ikl", Xd, Xd)                           # X_i'M_T X_i
    beta_i = ind.theta[:, 1:]
    eta_mg = beta_i - beta_i.mean(axis=0)
    Q = XMX.sum(axis=0) / (N * T)
    q = np.einsum("ikl,il->k", XMX, eta_mg) / (N * T)
    # h_{NT,beta} (80): H = s2_i X'MX/T  ->  s2 xfd'(X'MX/T)^-1 xfd
    h = float(np.mean([ind.sigma2[i] * xfd[i] @ np.linalg.solve(XMX[i] / T, xfd[i]) for i in range(N)]))
    if variant == "code":
        jk = half_jackknife(Y, W, ind.theta)[:, 1:]
        b = beta_i - jk
        eta = beta_i - fe.beta
        delta = float(((eta * xfd).sum(axis=1) ** 2).mean())
        cterm = 0.0
    elif variant == "paper":
        jk = _fe_half_jackknife(Y, W, fe.beta)
        b = np.tile(fe.beta - jk, (N, 1))
        delta = float(((eta_mg * xfd).sum(axis=1) ** 2).mean() - q @ np.linalg.solve(Q, q))
        ebar = ind.resid.mean(axis=1)
        cterm = float(np.mean(((fe.beta - beta_i) * xfd).sum(axis=1) * ebar))
    else:
        raise ValueError("variant must be 'code' or 'paper'")
    bx = (b * xfd).sum(axis=1)
    psi1 = T / N * (bx[:, None] * xfd).sum(axis=0)
    psi2 = T / N * (bx * (xfd * eta_mg).sum(axis=1)).sum()
    psi = float(psi1 @ np.linalg.solve(Q, q) - psi2)
    raw = (delta - psi / T - cterm) / (delta + h / T - 2 * psi / T)
    return WeightResult(_clip01(raw), float(raw),
                        {"Delta": delta, "h": h, "psi": psi, "c": cterm, "T": T})


@dataclass
class UnitWeights:
    """Unit-specific combination weights omega_i*, eq. (36)/(46)."""

    weights: np.ndarray        # (N,) weight on the individual forecast
    raw: np.ndarray            # before clipping (bias-adjusted variant only)
    omega_eta: np.ndarray
    sigma2: np.ndarray


def weights_unit_specific(Y: np.ndarray, W: np.ndarray, w_next: np.ndarray,
                          ind: IndividualFit, sigma: str = "ppt2022",
                          bias_adjusted: bool = False, ddof: int = 1) -> UnitWeights:
    """Unit-specific weights omega_i* = w'Om w / w'(T^-1 s_i^2 Q_iT^-1 + Om) w, eq. (36), (46).

    Reproduces ``panelRandomCoeffForecastsOldWeights.m`` (the "Comb. (omega_i*)"
    column of Tables 1-3): Om = cov(theta_hat_i) with ``ddof=1`` and, with
    ``sigma='ppt2022'``, s_i^2 = ||y_i - W_i theta_MG||^2 / (T + N^-1 sum_j w_j'(W_j'W_j)^-1 w_j)
    as in Pesaran, Pick and Timmermann (2022). ``sigma='ols'`` uses the OLS
    residual variance with T - K. ``bias_adjusted=True`` uses
    Om_C = Om - N^-1 sum s_i^2 (W_i'W_i)^-1 and clips to [0, 1] (code method 2).
    """
    N, T, K = W.shape
    om = _omega_eta(ind.theta, ddof=ddof)
    WWinv = np.stack([np.linalg.pinv(ind.WW[i]) for i in range(N)])
    if sigma == "ppt2022":
        tmg = ind.theta.mean(axis=0)
        res = Y - np.einsum("itk,k->it", W, tmg)
        ex = np.mean([w_next[i] @ np.linalg.inv(ind.WW[i]) @ w_next[i] for i in range(N)])
        s2 = (res ** 2).sum(axis=1) / (T + ex)
    elif sigma == "ols":
        s2 = ind.sigma2
    else:
        raise ValueError("sigma must be 'ppt2022' or 'ols'")
    if bias_adjusted:
        om_use = om - np.einsum("i,ikl->kl", s2, WWinv) / N
    else:
        om_use = om
    num = np.einsum("ik,kl,il->i", w_next, om_use, w_next)
    den = num + np.einsum("ik,ikl,il->i", w_next, s2[:, None, None] * WWinv, w_next)
    raw = num / den
    wts = np.clip(raw, 0.0, 1.0) if bias_adjusted else raw
    return UnitWeights(wts, raw, om, s2)


def weight_mallows(Y: np.ndarray, W: np.ndarray, ind: IndividualFit) -> WeightResult:
    """Mallows pooling-averaging weight on the POOLED estimator (Wang, Zhang and Paap, 2019).

    EXTENSION. Two-candidate case (pooled, individual) of their criterion (16)
    with A = X'X: C(w) = w^2 D + 2[w sum s_i^2 tr(S^-1 A_i) + (1-w) K sum s_i^2],
    D = sum (theta_pool - theta_i)'A_i(theta_pool - theta_i), S = sum A_i. The
    minimiser is w* = sum s_i^2 [K - tr(S^-1 A_i)] / D, clipped to [0, 1].
    """
    N, T, K = W.shape
    S = ind.WW.sum(axis=0)
    tp = np.linalg.solve(S, ind.Wy.sum(axis=0))
    d = tp - ind.theta
    D = float(np.einsum("ik,ikl,il->", d, ind.WW, d))
    tr = np.array([np.trace(np.linalg.solve(S, ind.WW[i])) for i in range(N)])
    raw = float((ind.sigma2 * (K - tr)).sum() / D) if D > 0 else 1.0
    return WeightResult(_clip01(raw), raw, {"D": D})


def weight_stein(Y: np.ndarray, W: np.ndarray, ind: IndividualFit, theta_pool: np.ndarray) -> WeightResult:
    """Stein-rule shrinkage weight on the POOLED estimator (Maddala et al., 1997).

    EXTENSION. weight = c / F, F the Chow statistic for H0: theta_1 = ... =
    theta_N with (q, N(T-K)) degrees of freedom, q = (N-1)K, and the
    Judge-Bock constant c = [(q-2)/q][N(T-K)/(N(T-K)+2)]. (Wang et al., 2019,
    write c/F as nu/F with nu = (q-2)/(NT-NK+2) and F unnormalised.) Clipped to [0, 1].
    """
    N, T, K = W.shape
    rss_ind = float((ind.resid ** 2).sum())
    rss_pool = float(((Y - np.einsum("itk,k->it", W, theta_pool)) ** 2).sum())
    df1, df2 = (N - 1) * K, N * T - N * K
    F = ((rss_pool - rss_ind) / df1) / (rss_ind / df2)
    nu = (df1 - 2) / df1 * df2 / (df2 + 2)
    raw = nu / F if F > 0 else 1.0
    return WeightResult(_clip01(raw), float(raw), {"F": float(F), "nu": float(nu)})


def oracle_weights(W: np.ndarray, w_next: np.ndarray, theta_true: np.ndarray,
                   eps: np.ndarray) -> Dict[str, float]:
    """Infeasible oracle weights using true parameters and disturbances (Table S.4).

    Mirrors ``panelRandomCorrCoeffFreqForecasts_wOracle.m``. ``theta_true`` is
    (N, K); ``eps`` is (N, T).
    """
    N, T, K = W.shape
    eta = theta_true - theta_true.mean(axis=0)
    WW = np.einsum("itk,itl->ikl", W, W)
    QN = WW.sum(axis=0) / (T * N)
    qN = np.einsum("ikl,il->k", WW, eta) / (T * N)
    D = psi1 = psi2 = hN = 0.0
    psi1 = np.zeros(K)
    X = W[:, :, 1:]
    xm = X.mean(axis=1)
    Xd = X - xm[:, None, :]
    xfd = w_next[:, 1:] - xm
    XMX = np.einsum("itk,itl->ikl", Xd, Xd)
    QFE, qFE = XMX.sum(axis=0) / (T * N), np.einsum("ikl,il->k", XMX, eta[:, 1:]) / (T * N)
    DFE = hFE = cFE = cNT = 0.0
    psiFE1 = np.zeros(K - 1)
    psiFE2 = 0.0
    cFE2 = np.zeros(K - 1)
    for i in range(N):
        wf = w_next[i]
        D += (wf @ eta[i]) ** 2 / N
        a = eps[i] @ W[i] @ np.linalg.inv(WW[i])
        psi1 = psi1 + T / N * (a @ wf) * wf
        psi2 += T / N * (a @ wf) * (wf @ eta[i])
        iQ = np.linalg.inv(WW[i] / T)
        g = W[i].T @ eps[i]
        hN += wf @ iQ @ np.outer(g, g) / T @ iQ @ wf / N
        e_b = eta[i, 1:]
        DFE += (xfd[i] @ e_b) ** 2 / N
        bfemb = e_b + eps[i] @ Xd[i] @ np.linalg.inv(XMX[i])
        psiFE1 = psiFE1 + T / N * (bfemb @ xfd[i]) * xfd[i]
        psiFE2 += T / N * (bfemb @ xfd[i]) * (xfd[i] @ e_b)
        iQF = np.linalg.inv(XMX[i] / T)
        gF = Xd[i].T @ eps[i]
        hFE += xfd[i] @ iQF @ np.outer(gF, gF) / T @ iQF @ xfd[i] / N
        cFE += (e_b @ xfd[i]) * eps[i].mean() / N
        cFE2 = cFE2 + xm[i] * eps[i].mean() / N
        cNT += (eps[i] @ Xd[i] @ np.linalg.inv(XMX[i]) @ xfd[i]) * eps[i].mean() / N
    D = max(D - qN @ np.linalg.solve(QN, qN), 0.0)
    psi = psi1 @ np.linalg.solve(QN, qN) - psi2
    DFE = max(DFE - qFE @ np.linalg.solve(QFE, qFE), 0.0)
    psiFE = psiFE1 @ np.linalg.solve(QFE, qFE) - psiFE2
    cFE = -cFE + qFE @ np.linalg.solve(QFE, cFE2)
    w_pool = _clip01((D - psi / T) / (D + hN / T - 2 * psi / T))
    w_fe = _clip01((DFE - psiFE / T - (cFE - cNT)) / (DFE + hFE / T - 2 * psiFE / T))
    return {"pool": w_pool, "fe": w_fe}


def check_weights_finite(*vals):
    for v in vals:
        if not np.isfinite(v):
            warnings.warn("A combination weight is not finite; check for collinear regressors.",
                          RuntimeWarning, stacklevel=3)
