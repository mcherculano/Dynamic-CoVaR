"""Data for the proof of concept: the bundled JPMorgan data set (load_jpmorgan), or public data
for any listed institution (load).

* Institution: daily total returns (dividends and splits) from Yahoo Finance (yfinance).
* Financial system: value-weighted daily return of the finance industry ("Money") among the
  12 industry portfolios of the Kenneth French data library. Unlike in the paper, the
  institution is not removed from the system portfolio, which public data do not allow.
* State variables (Adrian and Brunnermeier, 2016), lagged one trading day: changes in the
  3-month Treasury bill rate and in the 10-year minus 3-month slope (FRED DTB3, DGS10), change
  in the Baa minus 10-year credit spread (FRED BAA10Y, from 1986), and the market return and its
  22-day realised volatility (Kenneth French). The TED spread used in the paper is not included
  because LIBOR was discontinued in 2022.
Replace any of these with your own (for example, supervisory) data: the estimator only needs a
DataFrame with columns 'inst', 'sys' and the state variables.
"""
import io
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


@lru_cache(maxsize=None)
def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120).read()


def french_daily(name):
    """First table (value-weighted returns, %) of a daily file of the Kenneth French library."""
    with zipfile.ZipFile(io.BytesIO(_get(FRENCH.format(name)))) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))      # header row
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()), len(lines))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.columns = df.columns.str.strip()
    return df


def fred(series):
    df = pd.read_csv(io.BytesIO(_get(FRED.format(series))), index_col=0, parse_dates=True, na_values=".")
    return df.iloc[:, 0].rename(series)


def stock_returns(ticker):
    import yfinance as yf
    px = yf.download(ticker, start="1970-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    return (100 * px.pct_change()).dropna().rename("inst")


def load(ticker):
    """Daily data set for any listed institution, from public sources: institution and
    system returns (%) and lagged state variables."""
    return pd.concat([stock_returns(ticker), system_and_state()], axis=1, join="inner").dropna()


def load_jpmorgan():
    """The bundled JPMorgan data set (data/jpmorgan.csv): CRSP daily returns of JPMorgan Chase
    and its predecessors, extended with Yahoo Finance after 2021; quarterly balance-sheet
    characteristics from Compustat (leverage, maturity mismatch, log market-to-book), each
    used from its report date; system return and state variables as in system_and_state()."""
    return pd.read_csv(DATA / "jpmorgan.csv", index_col=0, parse_dates=True)


def system_and_state():
    """Financial-system return (%) and state variables lagged one trading day, public sources."""
    ff = french_daily("F-F_Research_Data_Factors_daily")
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("mkt")
    sys = french_daily("12_Industry_Portfolios_daily")["Money"].rename("sys")
    rates = pd.concat([fred(s) for s in ["DTB3", "DGS10", "BAA10Y"]], axis=1).reindex(mkt.index).ffill(limit=5)
    M = pd.DataFrame({
        "d_3m": rates["DTB3"].diff(),
        "d_slope": (rates["DGS10"] - rates["DTB3"]).diff(),
        "d_credit": rates["BAA10Y"].diff(),
        "mkt": mkt,
        "eqvol": mkt.rolling(22).std(),
    }).shift(1)                                           # state variables known at t-1
    return pd.concat([sys, M], axis=1).dropna()
