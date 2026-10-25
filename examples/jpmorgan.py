"""Dynamic CoVaR for JPMorgan Chase, daily, January 1986 to July 2024 (data/jpmorgan.csv).

    python examples/jpmorgan.py            # about one minute
    python examples/jpmorgan.py --q 0.01   # 1% quantile

Writes figures/jpmorgan_*.png and results/jpmorgan_*.csv.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dynamic_covar import data, run
from dynamic_covar.plotting import BLACK, BLUE, RED, band, save

RESULTS = Path(__file__).resolve().parents[1] / "results"
LABELS = {"inst": "Tail dependence on JPMorgan ($\\beta_t$)", "d_3m": "Change in 3-month yield",
          "d_slope": "Change in yield-curve slope", "d_credit": "Change in credit spread",
          "mkt": "Market return", "eqvol": "Equity volatility", "lev": "Leverage",
          "mm": "Maturity mismatch", "mb": "Market-to-book (log)"}
CRISES = [("1987-10-01", "1987-12-31"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]


def shade(ax):
    for a, b in CRISES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#c9d6ea", alpha=0.55, lw=0, zorder=0)


def main(q):
    df = data.load_jpmorgan()
    print(f"{len(df):,} daily observations, {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}")
    out = run(df, q)
    print("selected drift kappa:", out["kappa"])
    tag = f"q{round(100 * q):02d}"
    RESULTS.mkdir(exist_ok=True)
    out["results"].to_csv(RESULTS / f"jpmorgan_{tag}.csv")
    out["coef"].to_csv(RESULTS / f"jpmorgan_{tag}_coefficients.csv")

    monthly = out["results"].resample("ME").mean()
    for static, dyn, title, fname in [("covar", "dcovar", "CoVaR", "covar"),
                                      ("delta_covar", "delta_dcovar", "$\\Delta$CoVaR", "delta_covar")]:
        fig, ax = plt.subplots(figsize=(10, 4))
        shade(ax)
        ax.plot(monthly[static], color=BLUE, lw=1.1, label=f"Static {title}")
        band(ax, monthly[dyn], monthly[dyn + "_sd"], RED, f"Dynamic {title} (90% band)")
        ax.set_ylabel("Daily return of the financial system (%)")
        ax.set_title(f"JPMorgan: {title}, q = {q:.0%} (monthly averages of daily estimates)", loc="left")
        ax.legend(frameon=False, loc="lower left")
        save(fig, f"jpmorgan_{fname}_{tag}.png")

    coef, coef_sd = out["coef"].resample("ME").mean(), out["coef_sd"].resample("ME").mean()
    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    for ax, (name, label) in zip(axes.ravel(), LABELS.items()):
        shade(ax)
        band(ax, coef[name], coef_sd[name], BLACK, "Time-varying")
        ax.axhline(out["static_coef"][name], color=BLUE, ls="--", lw=1, label="Static")
        ax.set_title(label, loc="left", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"JPMorgan: coefficients of the system equation, q = {q:.0%} (90% bands)", x=0.01, ha="left")
    save(fig, f"jpmorgan_coefficients_{tag}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q", type=float, default=0.05)
    main(p.parse_args().q)
