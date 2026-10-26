"""
run_analysis -- NIFTY 50 Monte Carlo report.

    python run_analysis.py                         # latest snapshot, defaults
    python run_analysis.py --snapshot 2026-10-07   # a specific dated snapshot
    python run_analysis.py --history nifty.csv     # add a walk-forward VaR backtest
    python run_analysis.py --option-chain chain.csv  # compare flat-VIX model to market

Option-chain CSV columns: type (call/put), strike, and either price (premium
in index points) or iv (percent).  History CSV columns: Date, Close.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np

import nifty_mc as nm

HERE = Path(__file__).resolve().parent


def read_option_chain(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): v.strip() for k, v in r.items()}
            row = {"kind": r["type"].lower(), "strike": float(r["strike"])}
            row["price"] = float(r["price"]) if r.get("price") else None
            row["iv"] = float(r["iv"]) / 100.0 if r.get("iv") else None
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", help="as_of date (YYYY-MM-DD) in market_snapshots.json; default latest")
    ap.add_argument("--today", help="override today's date (YYYY-MM-DD), for reproducible staleness checks")
    ap.add_argument("--paths", type=int, default=1_000_000, help="paths for option pricing")
    ap.add_argument("--touch-paths", type=int, default=100_000, help="paths for touch probabilities")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history", help="daily Close CSV for the VaR backtest")
    ap.add_argument("--option-chain", help="option chain CSV (type, strike, price|iv)")
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    mkt = nm.load_snapshot(args.snapshot)
    F, r, sig, T, S0 = mkt.futures, mkt.risk_free, mkt.sigma, mkt.T, mkt.spot
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    age = (today - mkt.as_of).days
    seed = args.seed

    print("=" * 78)
    print(f"NIFTY 50 MONTE CARLO | snapshot: close of {mkt.as_of} | expiry {mkt.expiry}")
    print("=" * 78)
    print(f"Spot {S0:,.2f} | Futures {F:,.2f} (basis {mkt.basis:+.2f} pts, {mkt.basis / S0:+.3%}) | "
          f"India VIX {mkt.india_vix:.2f} | r {r:.2%} | lot {mkt.lot_size}")
    print(f"Time to expiry: {mkt.calendar_days} calendar days = {T:.5f} years (ACT/365)")
    if age > 3:
        print(f"WARNING: this snapshot is {age} days old (today = {today}). "
              f"Treat all output as a historical snapshot; add a newer entry to market_snapshots.json.")
    print("Volatility: ONE flat India VIX for every strike (a proxy, not each option's implied vol).")
    print("Sources: " + mkt.sources["spot_futures_vix"])
    print()

    # ---------------------------------------------------------------- 1. pricing
    atm = round(F / 50) * 50
    strikes = [atm + d for d in (-500, -250, 0, 250, 500)]
    band = 0.02
    print(f"1) EUROPEAN OPTIONS | Monte Carlo ({args.paths:,} paths, antithetic) vs Black-76")
    print(f"   Last two columns: Black-76 at VIX -/+ {band * 100:.0f} vol points = how much the flat-vol "
          f"assumption alone can move the price.")
    print(f"{'Type':<5}{'Strike':>7} | {'Black-76':>9}{'MC':>9}{'SE':>7}{'|err|/SE':>9} | "
          f"{'vol-2pt':>8}{'vol+2pt':>8}")
    print("-" * 78)
    for kind in ("call", "put"):
        for K in strikes:
            exact = nm.black76(F, K, r, sig, T, kind)
            mc, se = nm.mc_european(F, K, r, sig, T, args.paths, kind, np.random.default_rng(seed))
            lo = nm.black76(F, K, r, sig - band, T, kind)
            hi = nm.black76(F, K, r, sig + band, T, kind)
            print(f"{kind:<5}{K:>7.0f} | {exact:>9.2f}{mc:>9.2f}{se:>7.3f}{abs(mc - exact) / se:>9.2f} | "
                  f"{lo:>8.2f}{hi:>8.2f}")
    print("   |err|/SE below ~2 means the MC price agrees with the exact price.")

    chain = None
    if args.option_chain:
        chain = read_option_chain(args.option_chain)
        print(f"\n   MARKET vs MODEL ({len(chain)} quotes from {args.option_chain})")
        print(f"   {'Type':<5}{'Strike':>7} | {'Mkt price':>10}{'Mkt IV':>8} | {'Model@VIX':>10}{'Model-Mkt':>10}")
        for q in chain:
            if q["price"] is not None:
                try:
                    iv = nm.implied_vol(q["price"], F, q["strike"], r, T, q["kind"])
                except ValueError:
                    print(f"   {q['kind']:<5}{q['strike']:>7.0f} | {q['price']:>10.2f}{'n/a':>8} | "
                          f"skipped: price outside the no-arbitrage range for this snapshot's "
                          f"futures/expiry (stale or mismatched quote?)")
                    continue
                mkt_price = q["price"]
            else:
                iv = q["iv"]
                mkt_price = nm.black76(F, q["strike"], r, iv, T, q["kind"])
            model = nm.black76(F, q["strike"], r, sig, T, q["kind"])
            print(f"   {q['kind']:<5}{q['strike']:>7.0f} | {mkt_price:>10.2f}{iv * 100:>7.2f}% | "
                  f"{model:>10.2f}{model - mkt_price:>+10.2f}")
        print("   A varying Mkt IV across strikes is the volatility skew that a flat-VIX model ignores.")
    else:
        print("   No option chain supplied (--option-chain): no comparison with real premiums is made.")
    print()

    # ------------------------------------------------- 2. finishing / touch probs
    n_t = args.touch_paths
    days = mkt.calendar_days
    spd_main = 100
    spd_list = [1, 5, 25, 100, 400]
    up = [round(S0 * (1 + x) / 50) * 50 for x in (0.01, 0.02, 0.03)]
    dn = [round(S0 * (1 - x) / 50) * 50 for x in (0.01, 0.02, 0.03)]
    d2 = lambda K: (math.log(F / K) - 0.5 * sig**2 * T) / (sig * math.sqrt(T))

    steps_main = days * spd_main
    maxs, mins, terms = nm.simulate_path_extremes(S0, mkt.drift, sig, T, steps_main, n_t,
                                                  np.random.default_rng(seed))
    dt_main = T / steps_main
    print(f"2) WHERE CAN NIFTY GO BY {mkt.expiry}? ({n_t:,} paths, {steps_main:,} steps = "
          f"{spd_main}/calendar day, ACT/365 time)")
    print(f"{'Level':>8} | {'P(finish beyond)':>16}{'exact N(d2)':>12} | "
          f"{'P(touch) MC':>11}{'+/-SE':>7}{'analytic':>9}")
    print("-" * 78)
    for L in up + dn:
        above = L > S0
        fin = (terms > L).mean() if above else (terms < L).mean()
        ex_fin = nm.norm_cdf(d2(L)) if above else 1 - nm.norm_cdf(d2(L))
        p, se = nm.touch_prob_mc(maxs, mins, L, S0, sig, dt_main, bgk=True)
        an = nm.touch_prob_analytic(S0, L, mkt.drift, sig, T)
        print(f"{L:>7}{'+' if above else '-'} | {fin:>16.1%}{ex_fin:>12.1%} | {p:>11.1%}{se:>7.1%}{an:>9.1%}")
    print("   Risk-neutral probabilities (they price options; they are not forecasts).")
    print("   P(touch) MC uses the BGK barrier correction; 'analytic' is the exact continuous-monitoring value.")

    print(f"\n   MONITORING-FREQUENCY CONVERGENCE for the {up[1]}+ and {dn[1]}- levels")
    print(f"   {'steps/day':>9} | " + " | ".join(f"{('P(touch ' + str(L) + ')'):>26}" for L in (up[1], dn[1])))
    print(f"   {'':>9} | " + " | ".join(f"{'discrete':>8}{'BGK':>8}{'analytic':>9}" for _ in range(2)))
    conv = {}
    for spd in spd_list:
        st = days * spd
        mx, mn, _ = nm.simulate_path_extremes(S0, mkt.drift, sig, T, st, n_t,
                                              np.random.default_rng(seed + spd))
        cells = []
        for L in (up[1], dn[1]):
            raw, _ = nm.touch_prob_mc(mx, mn, L, S0, sig, T / st, bgk=False)
            cor, _ = nm.touch_prob_mc(mx, mn, L, S0, sig, T / st, bgk=True)
            an = nm.touch_prob_analytic(S0, L, mkt.drift, sig, T)
            conv[(spd, L)] = (raw, cor, an)
            cells.append(f"{raw:>8.1%}{cor:>8.1%}{an:>9.1%}")
        print(f"   {spd:>9} | " + " | ".join(cells))
    print("   Discrete monitoring understates touch probability and converges to the analytic value as steps\n"
          "   increase. The BGK correction removes most of the gap (it can overshoot slightly at 1 step/day).")
    print()

    # ------------------------------------------------------------------ 3. VaR
    units = mkt.lot_size
    notional = units * F
    print(f"3) MONTE CARLO VaR | long 1 lot Nifty futures (notional Rs {notional:,.0f}) | sigma = flat India VIX")
    print(f"{'Horizon':>8}{'Shocks':>12} | {'VaR 95%':>10}{'VaR 99%':>10}{'CVaR 99%':>10}{'exact 99%':>11}")
    print("-" * 66)
    pnl_store = {}
    for h in (1, 5):
        for label, df in (("Normal", None), ("Student-t4", 4)):
            pnl, v99, c99 = nm.mc_var(units, F, sig, h, 1_000_000, np.random.default_rng(seed), df=df, cl=0.99)
            _, v95, _ = nm.mc_var(units, F, sig, h, 1_000_000, np.random.default_rng(seed), df=df, cl=0.95)
            ex = nm.analytic_var_normal(units, F, sig, h, cl=0.99) if df is None else float("nan")
            pnl_store[(h, label)] = (pnl, v99)
            ex_s = f"{ex:>11,.0f}" if df is None else f"{'n/a':>11}"
            print(f"{h:>6}d {label:>12} | {v95:>10,.0f}{v99:>10,.0f}{c99:>10,.0f}{ex_s}")
    print("   These numbers describe the MODEL, not the real market: volatility is constant (no clustering,")
    print("   no jumps) and equal to the VIX proxy. Fat tails alone do not fix that. Accuracy is only")
    print("   tested if you supply history (--history), below.\n")

    if args.history:
        import var_backtest as vb
        dates, closes = vb.load_close_prices(args.history)
        rets = vb.log_returns(closes)
        hist_vol = rets.std(ddof=1) * math.sqrt(nm.TRADING_DAYS_PER_YEAR)
        print(f"4) HISTORY CHECK | {len(rets)} daily returns, {dates[0]} to {dates[-1]}")
        print(f"   Realised annualised volatility {hist_vol:.1%} vs India VIX proxy {mkt.india_vix:.1f}%")
        results = vb.backtest(rets, 0.99)
        vb.print_backtest(results, 0.99, f"history file: {Path(args.history).name}")
        print()
    else:
        print("4) HISTORY CHECK | skipped: no --history file. No claim is made about how accurate")
        print("   these VaR figures are on real Nifty returns. Run var_backtest.py --history <csv>.\n")

    print("LIMITS: flat volatility (no skew/smile), constant volatility (no GARCH/clustering), no jumps,")
    print("risk-neutral not real-world probabilities, VIX is a 30-day measure applied to a "
          f"{mkt.calendar_days}-day option, futures basis noise.\n")

    # ---------------------------------------------------------------- figure
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"NIFTY 50 Monte Carlo | snapshot {mkt.as_of} | expiry {mkt.expiry} | "
                     f"flat VIX {mkt.india_vix:.2f}% proxy", fontsize=13, fontweight="bold")

        # (a) paths
        demo_steps = 60
        rng = np.random.default_rng(3)
        dtp = T / demo_steps
        z = rng.standard_normal((40, demo_steps))
        paths = S0 * np.exp(np.cumsum((mkt.drift - 0.5 * sig**2) * dtp + sig * math.sqrt(dtp) * z, axis=1))
        paths = np.hstack([np.full((40, 1), S0), paths])
        tgrid = np.linspace(0, days, demo_steps + 1)
        for p in paths:
            ax[0, 0].plot(tgrid, p, lw=0.8, alpha=0.7)
        ax[0, 0].axhline(S0, color="k", lw=1, label=f"Spot {S0:,.0f}")
        ax[0, 0].axhline(F, color="red", lw=1, ls="--", label=f"Futures {F:,.0f}")
        ax[0, 0].set(title="40 simulated Nifty paths to expiry", xlabel="Calendar days from snapshot",
                     ylabel="Nifty")
        ax[0, 0].legend()

        # (b) touch convergence
        L = up[1]
        raw = [conv[(s, L)][0] for s in spd_list]
        cor = [conv[(s, L)][1] for s in spd_list]
        an = conv[(spd_list[0], L)][2]
        ax[0, 1].semilogx(spd_list, raw, "o-", label="discrete monitoring")
        ax[0, 1].semilogx(spd_list, cor, "s-", label="with BGK correction")
        ax[0, 1].axhline(an, color="red", ls="--", label=f"analytic continuous = {an:.1%}")
        ax[0, 1].set(title=f"P(touch {L}) vs monitoring frequency", xlabel="Time steps per calendar day",
                     ylabel="Probability")
        ax[0, 1].legend()

        # (c) pricing convergence
        K = atm
        exact = nm.black76(F, K, r, sig, T, "call")
        sizes = [10**k for k in (2, 3, 4, 5, 6)]
        res = [nm.mc_european(F, K, r, sig, T, s, "call", np.random.default_rng(seed)) for s in sizes]
        ax[1, 0].errorbar(sizes, [x[0] for x in res], yerr=[1.96 * x[1] for x in res], fmt="o-",
                          capsize=4, label="Antithetic MC (95% CI)")
        ax[1, 0].axhline(exact, color="red", ls="--", label=f"Black-76 = {exact:.2f}")
        ax[1, 0].set_xscale("log")
        ax[1, 0].set(title=f"Convergence: Nifty {K:.0f} CE", xlabel="Number of paths",
                     ylabel="Option price (index points)")
        ax[1, 0].legend()

        # (d) flat-vol sensitivity band
        ks = np.arange(atm - 750, atm + 751, 50)
        mid = [nm.black76(F, k, r, sig, T, "call") for k in ks]
        lo = [nm.black76(F, k, r, sig - band, T, "call") for k in ks]
        hi = [nm.black76(F, k, r, sig + band, T, "call") for k in ks]
        ax[1, 1].plot(ks, mid, label=f"Black-76 at VIX {mkt.india_vix:.1f}%")
        ax[1, 1].fill_between(ks, lo, hi, alpha=0.25, label=f"VIX -/+ {band * 100:.0f} vol points")
        if chain:
            cp = [(q["strike"], q["price"]) for q in chain if q["kind"] == "call" and q["price"] is not None]
            if cp:
                ax[1, 1].scatter(*zip(*cp), color="k", zorder=3, label="market calls")
        ax[1, 1].set(title="Call price vs strike: flat-vol sensitivity", xlabel="Strike",
                     ylabel="Option price (index points)")
        ax[1, 1].legend()

        fig.tight_layout()
        out = Path(args.outdir) / "nifty_mc_results.png"
        fig.savefig(out, dpi=140)
        print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
