"""Simulation exercises.

1. Why time-varying parameters: without state variables the CoVaR of Adrian and
   Brunnermeier (2016) is constant, while the Dynamic CoVaR tracks a moving true CoVaR.
2. Recovery of drifting quantile-regression coefficients, and no spurious drift when the
   coefficients are constant (small Monte Carlo).

    python examples/simulation.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from dynamic_covar import dynamic_covar, fit, static_covar
from dynamic_covar.plotting import BLUE, GREY, RED, band, save
from dynamic_covar.vb import ald_constants


def rmse(estimate, truth, axis=None):
    return np.sqrt(np.mean((estimate - truth) ** 2, axis=axis))


def ar1(n, phi, mean, sd, rng, start=None):
    x = np.empty(n); x[0] = mean if start is None else start
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + sd * rng.standard_normal()
    return x


def constant_covar_figure(q=0.05, T=1000, seed=11):
    """X^i = 0.6 M + e^i,  X^S = 0.4 M + beta_t X^i + e^S,  beta_t = 0.5 + 0.3 sin(2 pi t / T)."""
    rng = np.random.default_rng(seed)
    M = ar1(T + 1, 0.97, 0.0, np.sqrt(1 - 0.97 ** 2), rng)[1:]
    beta = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(T) / T)
    x_inst = 0.6 * M + rng.standard_normal(T)
    x_sys = 0.4 * M + beta * x_inst + rng.standard_normal(T)
    z = norm.ppf(q)
    true = 0.4 * M + beta * (0.6 * M + z) + z
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, state, title in [(axes[0], np.zeros((T, 0)), "(a) Without state variables"),
                             (axes[1], M, "(b) With the state variable")]:
        s, d = static_covar(x_inst, x_sys, state, q), dynamic_covar(x_inst, x_sys, state, q)
        ax.plot(true, color=GREY, ls="--", lw=1, label="True CoVaR")
        ax.plot(s["covar"], color=BLUE, lw=1.3, label="CoVaR")
        ax.plot(d["covar"], color=RED, lw=1.3, label="Dynamic CoVaR")
        ax.set_title(title, loc="left")
        print(f"{title}: RMSE CoVaR {rmse(s['covar'], true):.3f}, Dynamic CoVaR {rmse(d['covar'], true):.3f}")
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    axes[1].set_xlabel("t")
    save(fig, "simulation_constant_covar.png")


def ald_errors(q, T, rng):
    """Asymmetric Laplace errors with q-quantile zero and scale q(1-q)."""
    theta, phi2 = ald_constants(q)
    z = q * (1 - q) * rng.exponential(size=T)
    return theta * z + np.sqrt(phi2 * q * (1 - q) * z) * rng.standard_normal(T)


def coefficient_recovery(q=0.05, T=1000, reps=50, seed=1):
    """Intercept and slope follow AR(1) paths (persistence 0.99) with innovation sd s."""
    rng = np.random.default_rng(seed)
    print(f"\nCoefficient recovery, q = {q}, T = {T}, {reps} replications (RMSE: classic QR vs VB)")
    for s in [0.0, 0.02, 0.05]:
        err, kappas = np.zeros((reps, 2, 2)), []
        for r in range(reps):
            X = np.column_stack([np.ones(T), rng.standard_normal(T)])
            b = np.column_stack([ar1(T, 0.99, 0.0, s, rng), ar1(T, 0.99, 1.0, s, rng)])
            y = np.einsum("tp,tp->t", X, b) + ald_errors(q, T, rng)
            f = fit(y, X, q)
            err[r] = rmse(f["qr_coef"], b, axis=0), rmse(f["m"], b, axis=0)
            kappas.append(f["kappa"])
            if s == 0.05 and r == 0:
                example = (b, f)
        e = err.mean(0)
        print(f"  s = {s:.2f}: intercept {e[0, 0]:.3f} vs {e[1, 0]:.3f}, slope {e[0, 1]:.3f} vs {e[1, 1]:.3f}"
              f"  (median kappa {np.median(kappas)})")
    b, f = example
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, ax in enumerate(axes):
        ax.plot(b[:, j], color=GREY, lw=1.2, label="True coefficient")
        band(ax, f["m"][:, j], f["sd"][:, j], RED, "Variational posterior mean (90% band)", x=np.arange(T))
        ax.axhline(f["qr_coef"][j], color=BLUE, ls="--", lw=1, label="Classic quantile regression")
        ax.set_title(["Intercept", "Slope"][j], loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=10)
    save(fig, "simulation_coefficients.png", top=0.88)


if __name__ == "__main__":
    constant_covar_figure()
    coefficient_recovery()
