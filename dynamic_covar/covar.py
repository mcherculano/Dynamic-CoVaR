"""CoVaR (Adrian and Brunnermeier, 2016) and Dynamic CoVaR.

Two quantile regressions at level q, with state variables M_{t-1}:
    institution:  X^i_t     = alpha^i_t + gamma^i_t' M_{t-1} + e^i_t
    system:       X^{S|i}_t = alpha^S_t + gamma^S_t' M_{t-1} + beta_t X^i_t + e^S_t
VaR_t   = alpha^i_t + gamma^i_t' M_{t-1}
CoVaR_t = alpha^S_t + gamma^S_t' M_{t-1} + beta_t VaR_t
Delta CoVaR_t = beta_t (VaR_{q,t} - VaR_{0.5,t})
The static CoVaR has constant coefficients; the Dynamic CoVaR lets all of them follow
random walks and is estimated by variational Bayes (dynamic_covar.vb).
"""
import numpy as np
import pandas as pd

from .vb import fit, posterior_variance, quantile_regression


def _designs(x_inst, M):
    """Regressors of the institution and system equations (state variables standardised,
    so coefficients measure the effect of a one-standard-deviation change)."""
    M = np.asarray(M, float).reshape(len(x_inst), -1)
    X1 = np.column_stack([np.ones(len(M)), (M - M.mean(0)) / M.std(0)])
    return X1, np.column_stack([X1, x_inst])


def static_covar(x_inst, x_sys, M, q=0.05):
    X1, X2 = _designs(x_inst, M)
    qr1, qr50, qr2 = (quantile_regression(y, X, k) for y, X, k in
                      [(x_inst, X1, q), (x_inst, X1, 0.5), (x_sys, X2, q)])
    var, var50, b = X1 @ qr1[0], X1 @ qr50[0], qr2[0]
    covar = np.column_stack([X1, var]) @ b
    return dict(var=var, covar=covar, delta_covar=b[-1] * (var - var50), coef=b, qr=(qr1, qr50, qr2))


def dynamic_covar(x_inst, x_sys, M, q=0.05, qr=None):
    """Dynamic CoVaR with 90% posterior bands that propagate the uncertainty of both stages
    (the regressions are estimated separately and their posteriors treated as independent).
    qr: optional classic quantile regressions (institution at q and 0.5, system at q)."""
    X1, X2 = _designs(x_inst, M)
    qr = qr or (None, None, None)
    f1, f50, f2 = (fit(y, X, k, qr=r) for y, X, k, r in
                   [(x_inst, X1, q, qr[0]), (x_inst, X1, 0.5, qr[1]), (x_sys, X2, q, qr[2])])
    var, var50 = (np.einsum("tp,tp->t", X1, f["m"]) for f in (f1, f50))
    v_var, v_var50 = posterior_variance(f1, X1), posterior_variance(f50, X1)
    beta, v_beta = f2["m"][:, -1], f2["sd"][:, -1] ** 2
    C = np.column_stack([X1, var])
    covar = np.einsum("tp,tp->t", C, f2["m"])
    v_covar = posterior_variance(f2, C) + (beta ** 2 + v_beta) * v_var
    D, v_D = var - var50, v_var + v_var50
    v_delta = beta ** 2 * v_D + v_beta * D ** 2 + v_beta * v_D
    return dict(var=var, covar=covar, covar_sd=np.sqrt(v_covar), delta_covar=beta * D,
                delta_covar_sd=np.sqrt(v_delta), coef=f2["m"], coef_sd=f2["sd"],
                kappa=dict(var=f1["kappa"], median=f50["kappa"], covar=f2["kappa"]))


def run(data, q=0.05):
    """Static and Dynamic CoVaR for a DataFrame with columns 'inst', 'sys' (returns, %) and
    the state variables (lagged one period), indexed by date.

    Returns a dict with
      results       DataFrame: covar, dcovar, dcovar_sd, delta_covar, delta_dcovar, delta_dcovar_sd
      coef, coef_sd time-varying coefficients of the system equation and their posterior sd
      static_coef   coefficients of the static system equation
      kappa         selected drift parameters
    """
    x_inst, x_sys, M = data["inst"].values, data["sys"].values, data.drop(columns=["inst", "sys"])
    s = static_covar(x_inst, x_sys, M.values, q)
    d = dynamic_covar(x_inst, x_sys, M.values, q, qr=s["qr"])
    names = ["const"] + list(M.columns) + ["inst"]
    results = pd.DataFrame({"covar": s["covar"], "dcovar": d["covar"], "dcovar_sd": d["covar_sd"],
                            "delta_covar": s["delta_covar"], "delta_dcovar": d["delta_covar"],
                            "delta_dcovar_sd": d["delta_covar_sd"]}, index=data.index)
    return dict(results=results,
                coef=pd.DataFrame(d["coef"], index=data.index, columns=names),
                coef_sd=pd.DataFrame(d["coef_sd"], index=data.index, columns=names),
                static_coef=pd.Series(s["coef"], index=names), kappa=d["kappa"])
