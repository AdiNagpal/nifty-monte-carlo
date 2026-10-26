"""Tests for nifty_mc and var_backtest. Run from the project folder:  pytest -q"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nifty_mc as nm  # noqa: E402
import var_backtest as vb  # noqa: E402


# ---------------------------------------------------------------- Black-76
def test_black76_textbook_value():
    # F=K=100, sigma=20%, T=1, r=0  ->  call = F * (2 N(0.1) - 1) = 7.9656
    assert nm.black76(100, 100, 0.0, 0.2, 1.0, "call") == pytest.approx(7.9656, abs=1e-3)


def test_black76_put_call_parity():
    F, K, r, s, T = 22745.0, 22800.0, 0.0557, 0.157, 6 / 365
    c = nm.black76(F, K, r, s, T, "call")
    p = nm.black76(F, K, r, s, T, "put")
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-8)


def test_implied_vol_roundtrip():
    F, K, r, T = 22745.0, 22950.0, 0.0557, 6 / 365
    for kind in ("call", "put"):
        price = nm.black76(F, K, r, 0.173, T, kind)
        assert nm.implied_vol(price, F, K, r, T, kind) == pytest.approx(0.173, abs=1e-6)


def test_implied_vol_rejects_impossible_price():
    with pytest.raises(ValueError):
        nm.implied_vol(1e9, 100, 100, 0.05, 0.5, "call")


def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.05, 0.5, 0.95, 0.99):
        assert nm.norm_cdf(nm.norm_ppf(p)) == pytest.approx(p, abs=1e-10)


# ---------------------------------------------------------- Monte Carlo price
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [22300, 22750, 23200])
def test_mc_price_matches_black76_within_4_standard_errors(kind, K):
    F, r, s, T = 22745.0, 0.0557, 0.157, 6 / 365
    exact = nm.black76(F, K, r, s, T, kind)
    mc, se = nm.mc_european(F, K, r, s, T, 400_000, kind, np.random.default_rng(1))
    assert abs(mc - exact) < 4 * se


def test_mc_standard_error_shrinks_like_inverse_sqrt_n():
    F, K, r, s, T = 22745.0, 22750.0, 0.0557, 0.157, 6 / 365
    _, se_small = nm.mc_european(F, K, r, s, T, 40_000, "call", np.random.default_rng(2), antithetic=False)
    _, se_big = nm.mc_european(F, K, r, s, T, 640_000, "call", np.random.default_rng(3), antithetic=False)
    assert se_small / se_big == pytest.approx(4.0, rel=0.15)  # 16x paths -> 4x smaller


def test_antithetic_reduces_variance():
    F, K, r, s, T = 22745.0, 22750.0, 0.0557, 0.157, 6 / 365
    _, se_plain = nm.mc_european(F, K, r, s, T, 200_000, "call", np.random.default_rng(4), antithetic=False)
    _, se_anti = nm.mc_european(F, K, r, s, T, 200_000, "call", np.random.default_rng(4), antithetic=True)
    assert se_anti < se_plain


def test_mc_put_call_parity():
    F, K, r, s, T = 22745.0, 22800.0, 0.0557, 0.157, 6 / 365
    c, _ = nm.mc_european(F, K, r, s, T, 400_000, "call", np.random.default_rng(5))
    p, _ = nm.mc_european(F, K, r, s, T, 400_000, "put", np.random.default_rng(5))
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=0.5)


# ------------------------------------------------------- touch probabilities
def test_touch_analytic_driftless_closed_form():
    # nu = 0  (drift = sigma^2/2): P(touch a) = 2 * (1 - N(a / (sigma sqrt T)))
    S0, sigma, T = 100.0, 0.2, 0.25
    level = 108.0
    a = math.log(level / S0)
    expected = 2 * (1 - nm.norm_cdf(a / (sigma * math.sqrt(T))))
    assert nm.touch_prob_analytic(S0, level, 0.5 * sigma**2, sigma, T) == pytest.approx(expected, rel=1e-12)


def test_touch_analytic_bounds():
    assert nm.touch_prob_analytic(100, 160, 0.05, 0.2, 0.05) < 1e-6
    assert nm.touch_prob_analytic(100, 100.0001, 0.05, 0.2, 1.0) > 0.99
    # touching is at least as likely as finishing beyond the level
    S0, L, mu, s, T = 100, 105, 0.05, 0.2, 0.25
    d2 = (math.log(S0 * math.exp(mu * T) / L) - 0.5 * s**2 * T) / (s * math.sqrt(T))
    assert nm.touch_prob_analytic(S0, L, mu, s, T) > nm.norm_cdf(d2)


@pytest.mark.parametrize("level", [105.0, 94.0])
def test_bgk_correction_moves_discrete_estimate_toward_analytic(level):
    S0, mu, s, T, steps, n = 100.0, 0.05, 0.2, 0.1, 25, 200_000
    mx, mn, _ = nm.simulate_path_extremes(S0, mu, s, T, steps, n, np.random.default_rng(6))
    exact = nm.touch_prob_analytic(S0, level, mu, s, T)
    raw, _ = nm.touch_prob_mc(mx, mn, level, S0, s, T / steps, bgk=False)
    cor, se = nm.touch_prob_mc(mx, mn, level, S0, s, T / steps, bgk=True)
    assert raw < exact                          # discrete monitoring understates
    assert abs(cor - exact) < abs(raw - exact)  # correction helps
    assert abs(cor - exact) < 4 * se + 0.003    # and lands close


def test_fine_monitoring_converges_to_analytic():
    S0, mu, s, T, level, n = 100.0, 0.05, 0.2, 0.1, 105.0, 200_000
    exact = nm.touch_prob_analytic(S0, level, mu, s, T)
    errs = []
    for steps in (5, 50, 500):
        mx, mn, _ = nm.simulate_path_extremes(S0, mu, s, T, steps, n, np.random.default_rng(7))
        p, _ = nm.touch_prob_mc(mx, mn, level, S0, s, T / steps, bgk=False)
        errs.append(exact - p)
    assert errs[0] > errs[1] > errs[2] > -0.005  # bias shrinks as steps grow


def test_terminal_distribution_has_futures_as_mean():
    S0, mu, s, T = 100.0, 0.03, 0.2, 0.5
    _, _, term = nm.simulate_path_extremes(S0, mu, s, T, 10, 400_000, np.random.default_rng(8))
    assert term.mean() == pytest.approx(S0 * math.exp(mu * T), rel=0.002)


# --------------------------------------------------------------------- VaR
def test_mc_var_matches_analytic_for_normal_shocks():
    for h in (1, 5):
        _, v, _ = nm.mc_var(65, 22745.0, 0.157, h, 1_000_000, np.random.default_rng(9), cl=0.99)
        assert v == pytest.approx(nm.analytic_var_normal(65, 22745.0, 0.157, h, cl=0.99), rel=0.02)


def test_student_t_has_same_variance_but_fatter_tail():
    pnl_n, v_n, _ = nm.mc_var(65, 22745.0, 0.157, 1, 1_000_000, np.random.default_rng(10), cl=0.999)
    pnl_t, v_t, _ = nm.mc_var(65, 22745.0, 0.157, 1, 1_000_000, np.random.default_rng(10), df=4, cl=0.999)
    assert pnl_t.std() == pytest.approx(pnl_n.std(), rel=0.05)
    assert v_t > v_n


# -------------------------------------------------------- backtest statistics
def test_kupiec_perfect_and_extreme_cases():
    lr, p = vb.kupiec_pof(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    n, x, pr = 1000, 20, 0.01
    ph = x / n
    expected = -2 * ((n - x) * math.log(1 - pr) + x * math.log(pr)
                     - (n - x) * math.log(1 - ph) - x * math.log(ph))
    lr, p = vb.kupiec_pof(x, n, pr)
    assert lr == pytest.approx(expected, rel=1e-9)
    assert p < 0.05
    assert vb.kupiec_pof(0, 500, 0.01)[1] < 0.05  # zero exceptions is also suspicious


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(11)
    iid = (rng.random(3000) < 0.02).astype(int)
    clustered = np.zeros(3000, dtype=int)
    for start in range(100, 3000, 300):  # same ~2% rate, but in runs of 6
        clustered[start:start + 6] = 1
    assert vb.christoffersen_independence(iid)[1] > 0.05
    assert vb.christoffersen_independence(clustered)[1] < 0.01


def test_var_forecasts_have_no_lookahead():
    r = vb.synthetic_garch_returns(n=1200, seed=3)
    r2 = r.copy()
    r2[1000:] = 0.5  # change the future drastically
    a = vb.var_forecasts(r)
    b = vb.var_forecasts(r2)
    for name in a:
        np.testing.assert_allclose(a[name][:1000], b[name][:1000], equal_nan=True)


def test_backtest_on_fat_tailed_garch_data():
    # Synthetic GARCH(1,1) with Student-t(6) shocks: normal-quantile VaR should
    # under-cover at 99% (about twice the target rate), FHS should not.
    r = vb.synthetic_garch_returns(n=4000, seed=7)
    res = {x["model"]: x for x in vb.backtest(r, 0.99)}
    assert res["rolling"]["p_pof"] < 0.01
    assert res["ewma"]["p_pof"] < 0.01
    assert res["fhs"]["p_pof"] > 0.05
    target = 0.01
    assert abs(res["fhs"]["rate"] - target) < abs(res["ewma"]["rate"] - target)
    assert all(x["n"] == 4000 - 250 for x in res.values())  # warm-up excluded


# ------------------------------------------------------------- data loaders
def test_snapshot_loader_latest_and_errors():
    m = nm.load_snapshot()
    assert str(m.as_of) == "2026-10-21"
    assert m.calendar_days == 6 and m.T == pytest.approx(6 / 365)
    assert m.spot == 22811.35 and m.futures == 22745.0 and m.india_vix == 15.70
    assert nm.load_snapshot("2026-10-07").spot == 22603.05
    with pytest.raises(ValueError):
        nm.load_snapshot("1999-01-01")


def test_csv_loader_handles_dates_and_commas(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("Date,Open,High,Low,Close\n"
                 "22-10-2026,1,1,1,\"22,687.65\"\n"
                 "21-10-2026,1,1,1,\"22,811.35\"\n", encoding="utf-8")
    dates, closes = vb.load_close_prices(f)
    assert [str(d) for d in dates] == ["2026-10-21", "2026-10-22"]  # sorted ascending
    assert closes.tolist() == [22811.35, 22687.65]
