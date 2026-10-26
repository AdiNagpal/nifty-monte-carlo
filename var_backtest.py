"""
var_backtest -- walk-forward VaR backtesting with Kupiec and Christoffersen tests.

Every forecast for day t uses ONLY returns up to day t-1 (no look-ahead).
Three one-day VaR models are compared at a chosen confidence level:
    rolling   : normal VaR with the trailing-window standard deviation
    ewma      : normal VaR with RiskMetrics EWMA volatility (lambda = 0.94)
    fhs       : filtered historical simulation (EWMA volatility x empirical
                quantile of past standardised returns -> keeps fat tails)

An "exception" is a day on which the return fell below -VaR.

Usage:
    python var_backtest.py --history nifty.csv          # your real data
    python var_backtest.py                              # SYNTHETIC self-test

The synthetic mode only demonstrates the test machinery on simulated GARCH
data. It says nothing about how well any model fits the real Nifty.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np

from nifty_mc import norm_ppf

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_close_prices(csv_path):
    """
    Read a CSV with a date column and a close column (case-insensitive;
    'Close' or 'Adj Close'). Returns (dates, closes) sorted by date.
    Accepts ISO dates (2026-10-21) or DD-MM-YYYY / DD-Mon-YYYY style.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        date_col = cols.get("date")
        close_col = cols.get("close") or cols.get("adj close")
        if date_col is None or close_col is None:
            raise ValueError(f"CSV needs 'Date' and 'Close' columns; found {list(cols)}")
        rows = []
        for r in reader:
            raw_close = r[close_col].replace(",", "").strip()
            if raw_close in ("", "null", "NaN", "nan"):
                continue
            rows.append((_parse_date(r[date_col]), float(raw_close)))
    rows.sort(key=lambda x: x[0])
    return [d for d, _ in rows], np.array([c for _, c in rows])


def _parse_date(s):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {s!r}")


def log_returns(closes):
    return np.diff(np.log(closes))


def synthetic_garch_returns(n=3000, seed=7, omega=1.62e-6, alpha=0.09, beta=0.89, df=6):
    """
    SYNTHETIC daily returns from a GARCH(1,1) with Student-t innovations.
    Unconditional daily volatility = sqrt(omega / (1 - alpha - beta)) ~ 0.9%.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df, n) * math.sqrt((df - 2) / df)
    var = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = math.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    return r


# --------------------------------------------------------------------------
# Volatility forecasts (forecast for day t uses returns[:t])
# --------------------------------------------------------------------------
def ewma_sigma(returns, lam=0.94, init_window=30):
    """sigma[t] = EWMA volatility forecast for day t, using returns[:t] only."""
    n = len(returns)
    s2 = np.empty(n)
    s2[0] = np.var(returns[:init_window])
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(s2)


def rolling_sigma(returns, window=250):
    """sigma[t] = std of returns[t-window:t]; NaN until `window` obs exist."""
    n = len(returns)
    out = np.full(n, np.nan)
    for t in range(window, n):
        out[t] = returns[t - window:t].std(ddof=1)
    return out


# --------------------------------------------------------------------------
# VaR forecasts -> boolean exception series
# --------------------------------------------------------------------------
def var_forecasts(returns, cl=0.99, window=250, lam=0.94, fhs_window=500, warmup=250):
    """
    Returns {model_name: VaR array} where VaR[t] is a positive loss threshold
    for day t (NaN before `warmup`). Zero-mean returns assumed.
    """
    z = -norm_ppf(1 - cl)  # positive, e.g. 2.326 for 99%
    s_roll = rolling_sigma(returns, window)
    s_ewma = ewma_sigma(returns, lam)
    n = len(returns)

    fhs = np.full(n, np.nan)
    std_resid = returns / s_ewma  # std_resid[i] uses sigma forecast made before day i
    for t in range(max(warmup, 50), n):
        lo = max(0, t - fhs_window)
        q = np.quantile(std_resid[lo:t], 1 - cl)  # negative number
        fhs[t] = -q * s_ewma[t]

    out = {"rolling": z * s_roll, "ewma": z * s_ewma, "fhs": fhs}
    for k in out:
        out[k][:warmup] = np.nan
    return out


# --------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------
def _chi2_sf_1(x):
    """Survival function of chi-square with 1 degree of freedom."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(exceptions: int, observations: int, p: float):
    """
    Kupiec (1995) proportion-of-failures test of unconditional coverage.
    H0: exception probability = p. LR ~ chi-square(1). Returns (LR, p_value).
    """
    x, n = exceptions, observations
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2.0 * (ll0 - ll1)
    return lr, _chi2_sf_1(lr)


def christoffersen_independence(hits):
    """
    Christoffersen (1998) test that exceptions are independent over time
    (H0: today's exception does not depend on yesterday's). LR ~ chi-square(1).
    Returns (LR, p_value). Low p-values mean exceptions cluster.
    """
    h = np.asarray(hits, dtype=int)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    tot = n00 + n01 + n10 + n11
    pi = (n01 + n11) / tot
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
           + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr = -2.0 * (ll0 - ll1)
    return lr, _chi2_sf_1(lr)


def backtest(returns, cl=0.99, **kwargs):
    """Run all models; returns a list of result dicts."""
    p = 1 - cl
    forecasts = var_forecasts(returns, cl, **kwargs)
    results = []
    for name, var in forecasts.items():
        valid = ~np.isnan(var)
        hits = (returns[valid] < -var[valid]).astype(int)
        n, x = int(valid.sum()), int(hits.sum())
        lr_pof, p_pof = kupiec_pof(x, n, p)
        lr_ind, p_ind = christoffersen_independence(hits)
        results.append(dict(model=name, n=n, exceptions=x, expected=p * n,
                            rate=x / n, lr_pof=lr_pof, p_pof=p_pof,
                            lr_ind=lr_ind, p_ind=p_ind, var=var, hits=hits, valid=valid))
    return results


def print_backtest(results, cl, label):
    p = 1 - cl
    print(f"VaR BACKTEST ({label}) | one-day VaR at {cl:.0%} | walk-forward, no look-ahead")
    print(f"{'Model':<9}{'Obs':>6}{'Exc':>5}{'Exp':>7}{'Rate':>8} | "
          f"{'Kupiec p':>9}{'Indep p':>9} | verdict (5% level)")
    print("-" * 78)
    for r in results:
        bad = []
        if r["p_pof"] < 0.05:
            bad.append("wrong exception rate")
        if r["p_ind"] < 0.05:
            bad.append("exceptions cluster")
        verdict = "not rejected" if not bad else "REJECTED: " + ", ".join(bad)
        print(f"{r['model']:<9}{r['n']:>6}{r['exceptions']:>5}{r['expected']:>7.1f}"
              f"{r['rate']:>8.2%} | {r['p_pof']:>9.3f}{r['p_ind']:>9.3f} | {verdict}")
    print(f"(Target exception rate {p:.1%}. 'not rejected' means no evidence of a problem in "
          f"this sample, not proof the model is right.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", help="CSV with Date and Close columns (daily Nifty closes)")
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--plot", help="save a PNG of returns vs VaR to this path")
    args = ap.parse_args()

    if args.history:
        dates, closes = load_close_prices(args.history)
        returns = log_returns(closes)
        label = f"history file {Path(args.history).name}: {len(returns)} daily returns, {dates[0]} to {dates[-1]}"
    else:
        returns = synthetic_garch_returns()
        label = "SYNTHETIC GARCH(1,1) data, NOT Nifty: tests the machinery only"
    results = backtest(returns, args.confidence)
    print_backtest(results, args.confidence, label)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(returns, lw=0.5, color="grey", label="daily return")
        for r in results:
            if r["model"] in ("rolling", "ewma"):
                ax.plot(-r["var"], lw=1.1, label=f"-VaR {r['model']} ({r['exceptions']} exc.)")
        ax.set(title=f"Returns vs {args.confidence:.0%} VaR ({'SYNTHETIC data, not Nifty' if not args.history else Path(args.history).name})",
               xlabel="Day", ylabel="Log return")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
