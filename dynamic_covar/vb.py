"""Variational Bayes for quantile regression with random-walk coefficients.

Model (asymmetric Laplace working likelihood in mixture form, Kozumi and Kobayashi, 2011):

    y_t = x_t' b_t + theta z_t + phi sqrt(sigma z_t) u_t,   u_t ~ N(0, 1)
    z_t ~ Exponential(mean sigma),   sigma ~ InvGamma(a0, b0)
    b_t = b_{t-1} + v_t,             v_t ~ N(0, diag(omega)),   b_1 ~ N(mu0, V0)

with theta = (1 - 2q) / (q (1 - q)) and phi^2 = 2 / (q (1 - q)), so that x_t' b_t is the
q-quantile of y_t. The state variances are omega_j = (kappa * s_j)^2, where s_j is the
sampling standard deviation of coefficient j over one year in a classic quantile regression;
kappa, the speed of drift, is chosen by maximising the evidence lower bound (ELBO).

Variational family g(b_{1:T}) prod_t g(z_t) g(sigma):
  g(b_{1:T})  Gaussian over the whole path: Kalman filter + Durbin-Koopman smoother on
              pseudo-observations, O(T p^2) per pass;
  g(z_t)      generalised inverse Gaussian GIG(1/2, a, b_t), moments in closed form;
  g(sigma)    inverse gamma.
Coordinate ascent is accelerated by squared extrapolation (Varadhan and Roland, 2008);
an extrapolated step is kept only if it does not lower the ELBO.
"""
import math

import numpy as np
from numba import njit
from scipy.special import digamma, exp1, gammaln
from statsmodels.regression.quantile_regression import QuantReg

KAPPA_GRID = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)


def ald_constants(q):
    """theta and phi^2 of the normal-exponential mixture for quantile level q."""
    return (1 - 2 * q) / (q * (1 - q)), 2 / (q * (1 - q))


# ----------------------------------------------------------------------------------------
# Kalman filter and smoother: y_t = x_t' b_t + e_t, e_t ~ N(0, R_t); random-walk b_t
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def kalman_smoother(y, R, X, omega, mu0, V0, C):
    """Smoothed means m_t = E[b_t | y], the log-likelihood of y, the posterior variances
    c_t' Var(b_t | y) c_t for the rows c_t of C, and the posterior variances of each
    coefficient. The scalar observation makes every update rank one."""
    T, p = X.shape
    a = np.empty((T, p)); P = np.empty((T, p, p))
    v = np.empty(T); F = np.empty(T); K = np.empty((T, p))
    at, Pt, loglik = mu0.copy(), V0.copy(), 0.0
    for t in range(T):                                   # filter
        a[t], P[t] = at, Pt
        Px = Pt @ X[t]
        F[t] = X[t] @ Px + R[t]
        v[t] = y[t] - X[t] @ at
        K[t] = Px / F[t]
        loglik -= 0.5 * (math.log(2 * math.pi * F[t]) + v[t] ** 2 / F[t])
        at = at + K[t] * v[t]
        for i in range(p):                               # P <- P - Px Px'/F + diag(omega)
            for j in range(p):
                Pt[i, j] -= Px[i] * Px[j] / F[t]
            Pt[i, i] += omega[i]
    m = np.empty((T, p)); cVc = np.empty(T); var = np.empty((T, p))
    r = np.zeros(p); N = np.zeros((p, p))
    for t in range(T - 1, -1, -1):                       # smoother (Durbin and Koopman, 2012)
        x, k = X[t], K[t]
        r = x * (v[t] / F[t]) + r - x * (k @ r)          # r_{t-1} = x v / F + L' r,  L = I - k x'
        Nk = N @ k
        c = 1 / F[t] + k @ Nk
        for i in range(p):                               # N_{t-1} = x x'/F + L' N L
            for j in range(p):
                N[i, j] += x[i] * x[j] * c - x[i] * Nk[j] - Nk[i] * x[j]
        m[t] = a[t] + P[t] @ r
        Pc = P[t] @ C[t]
        cVc[t] = C[t] @ Pc - Pc @ N @ Pc                 # c' (P - P N P) c
        PN = P[t] @ N
        for j in range(p):                               # diag(P - P N P)
            s = 0.0
            for k in range(p):
                s += PN[j, k] * P[t, k, j]
            var[t, j] = P[t, j, j] - s
    return m, loglik, cVc, var


# ----------------------------------------------------------------------------------------
# Closed-form moments
# ----------------------------------------------------------------------------------------
def _exp_e1(x):
    """e^x E1(x): exact below 50, four-term asymptotic series above (relative error < 1e-7)."""
    xs = np.minimum(x, 50.0)
    return np.where(x < 50, np.exp(xs) * exp1(xs), (1 - 1 / x + 2 / x ** 2 - 6 / x ** 3) / x)


def gig_half_moments(a, b):
    """E[z], E[1/z], E[log z] and E[log g(z)] for z ~ GIG(1/2, a, b),
    density proportional to z^{-1/2} exp(-(a z + b / z) / 2)."""
    w = np.sqrt(a * b)
    Ez = np.sqrt(b / a) * (1 + 1 / w)
    Einvz = np.sqrt(a / b)
    Elogz = 0.5 * np.log(b / a) + _exp_e1(2 * w)       # d/dp log K_p(w) at p = 1/2
    logK = 0.5 * np.log(np.pi / (2 * w)) - w             # log K_{1/2}(w)
    Elogg = 0.25 * np.log(a / b) - np.log(2) - logK - 0.5 * Elogz - 0.5 * (a * Ez + b * Einvz)
    return Ez, Einvz, Elogz, Elogg


def _invgamma_elbo_terms(alpha, beta, a0, b0):
    """E[log p(sigma)] - E[log g(sigma)] for prior IG(a0, b0) and g = IG(alpha, beta)."""
    Elog, Einv = np.log(beta) - digamma(alpha), alpha / beta
    return (a0 * np.log(b0) - gammaln(a0) - (a0 + 1) * Elog - b0 * Einv
            - (alpha * np.log(beta) - gammaln(alpha) - (alpha + 1) * Elog - alpha))


def quantile_regression(y, X, q):
    """Classic quantile regression; returns coefficients and their standard errors."""
    res = QuantReg(y, X).fit(q=q, max_iter=5000)
    return np.asarray(res.params), np.asarray(res.bse)


# ----------------------------------------------------------------------------------------
# Coordinate ascent
# ----------------------------------------------------------------------------------------
class _Model:
    def __init__(self, y, X, q, omega, mu0, V0, a0=0.01, b0=0.01):
        self.y, self.X, self.q, self.omega, self.mu0, self.V0 = y, X, q, omega, mu0, V0
        self.theta, self.phi2 = ald_constants(q)
        self.a0, self.b0, self.alpha = a0, b0, a0 + 1.5 * len(y)

    def update_z_sigma(self, resid, A, Einv_sig):
        """g(z_t) given g(b) and the current g(sigma), then g(sigma). Returns the new state
        th = (log b_1..log b_T, log a, log beta_sigma)."""
        a = Einv_sig * (self.theta ** 2 / self.phi2 + 2)
        b = Einv_sig * A / self.phi2
        Ez, Einvz, _, _ = gig_half_moments(a, b)
        beta = self.b0 + np.sum((Einvz * A - 2 * self.theta * resid + self.theta ** 2 * Ez) / (2 * self.phi2) + Ez)
        return np.append(np.log(b), [np.log(a), np.log(beta)])

    def initial(self, b_qr):
        """Starting state from the classic quantile regression (prior variance of the fit
        stands in for x' Var(b) x, which keeps A_t > 0 at interpolated points)."""
        resid = self.y - self.X @ b_qr
        Einv_sig = 1 / np.mean(resid * (self.q - (resid < 0)))
        A = resid ** 2 + np.einsum("tp,pq,tq->t", self.X, self.V0, self.X) * 1e-6
        return self.update_z_sigma(resid, A, Einv_sig)

    def sweep(self, th):
        """One exact coordinate-ascent sweep from state th. Returns the next state, the ELBO
        at th, and the moments of g(b_{1:T})."""
        y, X, theta, phi2 = self.y, self.X, self.theta, self.phi2
        b_z, a_z, beta = np.exp(th[:-2]), np.exp(th[-2]), np.exp(th[-1])
        Einv_sig, Elog_sig = self.alpha / beta, np.log(beta) - digamma(self.alpha)
        Ez, Einvz, Elogz, Elogg = gig_half_moments(a_z, b_z)
        ytil, R = y - theta / Einvz, phi2 / (Einv_sig * Einvz)   # pseudo-observations
        m, loglik, xVx, var = kalman_smoother(ytil, R, X, self.omega, self.mu0, self.V0, X)
        elbo = (loglik + 0.5 * np.sum(np.log(2 * np.pi * R))
                + np.sum(Einv_sig * theta ** 2 / (2 * phi2) * (1 / Einvz - Ez)
                         - 0.5 * Elogz - 1.5 * Elog_sig - 0.5 * np.log(2 * np.pi * phi2)
                         - Einv_sig * Ez - Elogg)
                + _invgamma_elbo_terms(self.alpha, beta, self.a0, self.b0))
        resid = y - np.einsum("tp,tp->t", X, m)
        th_next = self.update_z_sigma(resid, resid ** 2 + xVx, Einv_sig)
        return th_next, elbo, dict(m=m, sd=np.sqrt(var), ytil=ytil, R=R)


def _accelerated_ascent(model, th, tol=1e-7, max_sweeps=300):
    """Squared extrapolation (Varadhan and Roland, 2008): after two ordinary sweeps, jump
    along the path they trace; keep the jump only if the ELBO there is not lower than at
    the intermediate point, otherwise keep the two ordinary sweeps."""
    sweeps, elbo_prev = 0, -np.inf
    while sweeps < max_sweeps:
        th1, _, _ = model.sweep(th)
        th2, elbo1, _ = model.sweep(th1)
        rho, nu = th1 - th, th2 - 2 * th1 + th
        alpha = min(-np.linalg.norm(rho) / max(np.linalg.norm(nu), 1e-300), -1.0)
        th_x, elbo_x, _ = model.sweep(th - 2 * alpha * rho + alpha ** 2 * nu)
        sweeps += 3
        th, elbo = (th_x, elbo_x) if np.isfinite(elbo_x) and elbo_x >= elbo1 else (th2, elbo1)
        if abs(elbo - elbo_prev) < tol * abs(elbo):
            break
        elbo_prev = elbo
    return th


def fit_fixed_kappa(y, X, q, kappa, qr=None, init=None, periods_per_year=252):
    """Variational posterior at a given drift parameter kappa (scalar or one per coefficient)."""
    y, X = np.ascontiguousarray(y, float), np.ascontiguousarray(X, float)
    b_qr, se_qr = quantile_regression(y, X, q) if qr is None else qr
    s = se_qr * np.sqrt(len(y) / periods_per_year)
    model = _Model(y, X, q, (np.asarray(kappa, float) * s) ** 2, b_qr, np.diag((10 * s) ** 2))
    th = _accelerated_ascent(model, model.initial(b_qr) if init is None else init)
    _, elbo, post = model.sweep(th)
    return dict(post, elbo=elbo, kappa=kappa, theta=th, qr_coef=b_qr, model=model)


def fit(y, X, q, kappas=KAPPA_GRID, qr=None, periods_per_year=252):
    """Time-varying quantile regression. Fits the model over the kappa grid (warm starts),
    doubling kappa while the largest value has the highest ELBO (up to kappa = 5), and
    returns the fit with the highest ELBO together with the ELBO profile."""
    qr = quantile_regression(y, X, q) if qr is None else qr
    kappas, fits = list(kappas), []
    for k in kappas:
        fits.append(fit_fixed_kappa(y, X, q, k, qr, fits[-1]["theta"] if fits else None, periods_per_year))
    while fits[-1]["elbo"] == max(f["elbo"] for f in fits) and kappas[-1] < 5:
        kappas.append(2 * kappas[-1])
        fits.append(fit_fixed_kappa(y, X, q, kappas[-1], qr, fits[-1]["theta"], periods_per_year))
    best = max(fits, key=lambda f: f["elbo"])
    best["elbo_profile"] = {k: f["elbo"] for k, f in zip(kappas, fits)}
    return best


def posterior_variance(fit_, C):
    """Var(c_t' b_t) under g(b_{1:T}) for each row c_t of C."""
    mdl = fit_["model"]
    return kalman_smoother(fit_["ytil"], fit_["R"], mdl.X, mdl.omega, mdl.mu0, mdl.V0,
                           np.ascontiguousarray(C, float))[2]
