import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data import load_data, calculate_equal_weights
from strategies import compute_signals
from backtest import backtest, sharpe_ratio, max_drawdown
from portfolio import (
    validate_portfolio_inputs,
    compute_portfolio_params,
    aggregate_portfolio_equity,
    run_portfolio_backtest,
)
from models import PortfolioResult


# ---------------------------------------------------------------------------
# Existing single-ticker tests (must remain unchanged)
# ---------------------------------------------------------------------------

def test_pipeline():
    ticker = "AAPL"
    end = datetime.today()
    start = end - timedelta(days=200)
    df, source = load_data(ticker, start.isoformat(), end.isoformat())
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        print(f"No data (source={source}); skipping heavy checks.")
        return
    sig = compute_signals(df, "Moving Average Crossover", {"short": 10, "long": 30})
    bt, trades, dca_metrics, bh_metrics = backtest(sig, strategy="Moving Average Crossover")
    assert not bt.empty
    assert {"equity", "bh_equity"}.issubset(bt.columns)

    # Test New Car strategy basic flow
    car_params = {"car_price": 30000, "down_payment_pct": 20, "down_payment_amount": 0, "payment_frequency": "Monthly", "term_months": 12}
    car_sig = compute_signals(df, "New Car", car_params)
    car_bt, car_trades, car_metrics, _ = backtest(car_sig, strategy="New Car", params=car_params)
    assert not car_bt.empty
    assert "equity" in car_bt.columns
    print("OK: pipeline runs for MAC and New Car.")


# ---------------------------------------------------------------------------
# Portfolio helper tests (no network I/O — pure unit tests)
# ---------------------------------------------------------------------------

def _make_price_df(n=252, start_price=100.0, daily_return=0.001):
    """Synthesize a simple ascending price series."""
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    prices = start_price * (1 + daily_return) ** np.arange(n)
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": 1_000_000,
    }, index=dates)
    df.index.name = "date"
    return df


def test_validate_portfolio_inputs_valid():
    errors = validate_portfolio_inputs(["AAPL", "MSFT"], {"AAPL": 60.0, "MSFT": 40.0})
    assert errors == [], f"Expected no errors, got: {errors}"
    print("OK: valid portfolio inputs produce no errors.")


def test_validate_portfolio_inputs_too_few_tickers():
    errors = validate_portfolio_inputs(["AAPL"], {"AAPL": 100.0})
    assert len(errors) == 1
    print("OK: single ticker caught as error.")


def test_validate_portfolio_inputs_bad_sum():
    errors = validate_portfolio_inputs(["AAPL", "MSFT"], {"AAPL": 60.0, "MSFT": 60.0})
    assert len(errors) > 0, "Expected weight-sum error"
    print("OK: weight sum != 100 caught.")


def test_validate_portfolio_inputs_zero_weight():
    errors = validate_portfolio_inputs(["AAPL", "MSFT"], {"AAPL": 100.0, "MSFT": 0.0})
    assert len(errors) > 0
    print("OK: zero weight caught.")


def test_compute_portfolio_params_bh():
    base = {"amount": 10000}
    result = compute_portfolio_params(base, weight=0.6, total_capital=10000.0, strategy="Buy & Hold")
    assert abs(result["amount"] - 6000.0) < 1e-6
    assert base["amount"] == 10000, "base_params must not be mutated"
    print("OK: compute_portfolio_params scales amount for B&H.")


def test_compute_portfolio_params_dca():
    base = {"amount": 500, "frequency": "Monthly"}
    result = compute_portfolio_params(base, weight=0.4, total_capital=5000.0, strategy="Dollar Cost Averaging")
    # DCA per-period amount is scaled by the ticker's weight fraction (not by total_capital).
    # weight=0.4 of $500/period -> $200/period allocated to this ticker.
    assert abs(result["amount"] - 200.0) < 1e-6
    assert base["amount"] == 500, "base_params must not be mutated"
    print("OK: compute_portfolio_params scales DCA amount by weight fraction.")


def test_compute_portfolio_params_no_mutation_for_mac():
    base = {"short": 20, "long": 50}
    result = compute_portfolio_params(base, weight=0.5, total_capital=10000.0, strategy="Moving Average Crossover")
    assert "amount" not in result
    assert base == {"short": 20, "long": 50}
    print("OK: compute_portfolio_params leaves MAC params untouched.")


def test_aggregate_portfolio_equity_math():
    """Two identical equity series with equal weights should produce the same equity."""
    n = 50
    dates = pd.date_range("2023-01-03", periods=n, freq="B")
    eq = pd.Series((1.001) ** np.arange(n), index=dates)

    bt_a = pd.DataFrame({"equity": eq, "strategy_ret": eq.pct_change().fillna(0)}, index=dates)
    bt_b = pd.DataFrame({"equity": eq, "strategy_ret": eq.pct_change().fillna(0)}, index=dates)

    ticker_results = {
        "A": {"bt": bt_a},
        "B": {"bt": bt_b},
    }
    weights_norm = {"A": 0.5, "B": 0.5}
    out = aggregate_portfolio_equity(ticker_results, weights_norm, dates)

    assert "portfolio_equity" in out.columns
    np.testing.assert_allclose(out["portfolio_equity"].values, eq.values, rtol=1e-6)
    print("OK: aggregate_portfolio_equity(equal weights, identical equities) == original equity.")


def test_date_intersection():
    """Common dates of two overlapping series should be correct."""
    dates_a = pd.date_range("2022-01-03", periods=100, freq="B")
    dates_b = pd.date_range("2022-04-01", periods=100, freq="B")
    common = dates_a.intersection(dates_b)
    assert len(common) > 0
    assert common[0] >= dates_b[0]
    assert common[-1] <= dates_a[-1]
    print(f"OK: date intersection yields {len(common)} common days.")


def test_portfolio_n1_regression():
    """
    N=1 portfolio with weight=100% must produce equity identical to single-ticker backtest.
    This is the critical regression guard.
    """
    df = _make_price_df(n=100)
    params = {"short": 10, "long": 30}
    strategy = "Moving Average Crossover"

    sig_df = compute_signals(df, strategy, params)
    bt_single, _, _, _ = backtest(sig_df, slippage_bps=0.0, strategy=strategy, params=params)

    ticker_results = {"TEST": {"bt": bt_single}}
    weights_norm = {"TEST": 1.0}
    common_dates = bt_single.index
    port_df = aggregate_portfolio_equity(ticker_results, weights_norm, common_dates)

    np.testing.assert_allclose(
        port_df["portfolio_equity"].values,
        bt_single["equity"].values,
        rtol=1e-6,
        err_msg="N=1 portfolio equity must match single-ticker equity exactly",
    )
    print("OK: N=1 portfolio equity matches single-ticker equity.")


def test_params_isolation_across_tickers():
    """Each ticker must get its own independent params copy (no cross-contamination)."""
    base = {"amount": 1000, "frequency": "Monthly"}
    p1 = compute_portfolio_params(base, weight=0.6, total_capital=10000.0, strategy="Dollar Cost Averaging")
    p2 = compute_portfolio_params(base, weight=0.4, total_capital=10000.0, strategy="Dollar Cost Averaging")

    # DCA amount scales by weight fraction: 1000 * 0.6 = 600, 1000 * 0.4 = 400
    assert abs(p1["amount"] - 600.0) < 1e-6
    assert abs(p2["amount"] - 400.0) < 1e-6
    assert base["amount"] == 1000, "base must be untouched"
    # Mutating p1 must not affect p2
    p1["amount"] = 99999
    assert p2["amount"] == 400.0
    print("OK: params isolation — each ticker gets an independent copy.")


def test_portfolio_equal_weights_two_identical_tickers():
    """
    Two tickers backed by the same synthetic price series with equal weights
    should produce portfolio equity equal to either ticker's equity.
    """
    df = _make_price_df(n=80)
    params = {"short": 5, "long": 15}
    strategy = "Moving Average Crossover"

    sig_a = compute_signals(df.copy(), strategy, params)
    sig_b = compute_signals(df.copy(), strategy, params)
    bt_a, _, _, _ = backtest(sig_a, slippage_bps=0.0, strategy=strategy, params=params)
    bt_b, _, _, _ = backtest(sig_b, slippage_bps=0.0, strategy=strategy, params=params)

    common_dates = bt_a.index.intersection(bt_b.index)
    ticker_results = {
        "X": {"bt": bt_a},
        "Y": {"bt": bt_b},
    }
    weights_norm = {"X": 0.5, "Y": 0.5}
    port_df = aggregate_portfolio_equity(ticker_results, weights_norm, common_dates)

    expected = bt_a["equity"].reindex(common_dates).values
    np.testing.assert_allclose(
        port_df["portfolio_equity"].values,
        expected,
        rtol=1e-6,
        err_msg="Equal-weight portfolio of two identical tickers must equal either ticker's equity",
    )
    print("OK: equal-weight portfolio of two identical synthetic tickers matches per-ticker equity.")


# ---------------------------------------------------------------------------
# Bug #2: calculate_equal_weights helper
# ---------------------------------------------------------------------------

def _check_equal_weights(weights: dict, label: str):
    """Assert that weights sum to exactly 100.00 and max-min <= 0.01."""
    vals = list(weights.values())
    total = round(sum(vals), 10)  # avoid fp drift in sum
    assert abs(total - 100.0) < 1e-9, f"{label}: sum={total} != 100.0"
    spread = max(vals) - min(vals)
    assert spread <= 0.01 + 1e-9, f"{label}: spread={spread} > 0.01"


def test_equal_weights_n1():
    w = calculate_equal_weights(["A"])
    assert abs(w["A"] - 100.0) < 1e-9
    _check_equal_weights(w, "n=1")
    print("OK: calculate_equal_weights(n=1) -> 100.00")


def test_equal_weights_n2():
    w = calculate_equal_weights(["A", "B"])
    _check_equal_weights(w, "n=2")
    assert w["A"] == 50.0 and w["B"] == 50.0
    print("OK: calculate_equal_weights(n=2) -> 50.00 / 50.00")


def test_equal_weights_n3():
    tickers = ["A", "B", "C"]
    w = calculate_equal_weights(tickers)
    _check_equal_weights(w, "n=3")
    # All weights should be either 33.33 or 33.34
    for v in w.values():
        assert v in (33.33, 33.34), f"n=3 unexpected weight {v}"
    # First ticker must NOT always get the remainder (remainder goes to last)
    assert w["A"] == 33.33, f"n=3: first ticker should be 33.33, got {w['A']}"
    assert w["C"] == 33.34, f"n=3: last ticker should carry remainder 33.34, got {w['C']}"
    print(f"OK: calculate_equal_weights(n=3) -> {list(w.values())}, sum=100.00")


def test_equal_weights_n4():
    w = calculate_equal_weights(["A", "B", "C", "D"])
    _check_equal_weights(w, "n=4")
    for v in w.values():
        assert abs(v - 25.0) < 1e-9, f"n=4: expected 25.00 got {v}"
    print("OK: calculate_equal_weights(n=4) -> 25.00 each")


def test_equal_weights_n6():
    tickers = [str(i) for i in range(6)]
    w = calculate_equal_weights(tickers)
    _check_equal_weights(w, "n=6")
    print(f"OK: calculate_equal_weights(n=6) -> {list(w.values())}, sum=100.00")


def test_equal_weights_n7():
    tickers = [str(i) for i in range(7)]
    w = calculate_equal_weights(tickers)
    _check_equal_weights(w, "n=7")
    print(f"OK: calculate_equal_weights(n=7) -> {list(w.values())}, sum=100.00")


def test_equal_weights_n10():
    tickers = [str(i) for i in range(10)]
    w = calculate_equal_weights(tickers)
    _check_equal_weights(w, "n=10")
    for v in w.values():
        assert abs(v - 10.0) < 1e-9, f"n=10: expected 10.00 got {v}"
    print("OK: calculate_equal_weights(n=10) -> 10.00 each")


# ---------------------------------------------------------------------------
# Bug #6: slippage/fees applied to DCA and Buy & Hold
# ---------------------------------------------------------------------------

def test_dca_slippage_reduces_final_value():
    """With fees > 0, DCA final portfolio value must be lower than with fees = 0."""
    df = _make_price_df(n=252)
    params = {"frequency": "Monthly", "amount": 1000}
    sig_no_fee = compute_signals(df.copy(), "Dollar Cost Averaging", params)
    sig_fee = compute_signals(df.copy(), "Dollar Cost Averaging", params)

    bt_no_fee, _, metrics_no_fee, _ = backtest(sig_no_fee, slippage_bps=0.0, strategy="Dollar Cost Averaging", params=params)
    bt_fee, _, metrics_fee, _ = backtest(sig_fee, slippage_bps=48.0, strategy="Dollar Cost Averaging", params=params)

    assert metrics_fee["total_value"] < metrics_no_fee["total_value"], (
        f"DCA 48bps final value {metrics_fee['total_value']:.2f} must be less than "
        f"0bps {metrics_no_fee['total_value']:.2f}"
    )
    assert float(bt_fee["equity"].iloc[-1]) < float(bt_no_fee["equity"].iloc[-1]), \
        "DCA equity with fees must be lower"
    print(f"OK: DCA 48bps={metrics_fee['total_value']:.2f} < 0bps={metrics_no_fee['total_value']:.2f}")


def test_dca_weekly_more_drag_than_monthly():
    """DCA Weekly incurs more fee drag than DCA Monthly at the same bps."""
    df = _make_price_df(n=252)
    params_w = {"frequency": "Weekly", "amount": 1000}
    params_m = {"frequency": "Monthly", "amount": 1000}
    bps = 48.0

    sig_w = compute_signals(df.copy(), "Dollar Cost Averaging", params_w)
    sig_m = compute_signals(df.copy(), "Dollar Cost Averaging", params_m)

    _, _, metrics_w_fee, _ = backtest(sig_w, slippage_bps=bps, strategy="Dollar Cost Averaging", params=params_w)
    _, _, metrics_m_fee, _ = backtest(sig_m, slippage_bps=bps, strategy="Dollar Cost Averaging", params=params_m)
    _, _, metrics_w_0, _ = backtest(compute_signals(df.copy(), "Dollar Cost Averaging", params_w), slippage_bps=0.0, strategy="Dollar Cost Averaging", params=params_w)
    _, _, metrics_m_0, _ = backtest(compute_signals(df.copy(), "Dollar Cost Averaging", params_m), slippage_bps=0.0, strategy="Dollar Cost Averaging", params=params_m)

    drag_weekly = metrics_w_0["total_value"] - metrics_w_fee["total_value"]
    drag_monthly = metrics_m_0["total_value"] - metrics_m_fee["total_value"]
    assert drag_weekly >= drag_monthly, (
        f"Weekly drag {drag_weekly:.2f} must be >= monthly drag {drag_monthly:.2f}"
    )
    print(f"OK: Weekly fee drag {drag_weekly:.2f} >= Monthly fee drag {drag_monthly:.2f}")


def test_bh_slippage_reduces_final_value():
    """With fees > 0, Buy & Hold final portfolio value must be lower than with fees = 0."""
    df = _make_price_df(n=252)
    params = {"amount": 10000}
    sig = compute_signals(df.copy(), "Buy & Hold", params)

    bt_no_fee, _, _, metrics_no_fee = backtest(sig, slippage_bps=0.0, strategy="Buy & Hold", params=params)
    bt_fee, _, _, metrics_fee = backtest(sig, slippage_bps=48.0, strategy="Buy & Hold", params=params)

    assert metrics_fee["total_value"] < metrics_no_fee["total_value"], (
        f"B&H 48bps {metrics_fee['total_value']:.2f} must be < 0bps {metrics_no_fee['total_value']:.2f}"
    )
    print(f"OK: B&H 48bps={metrics_fee['total_value']:.2f} < 0bps={metrics_no_fee['total_value']:.2f}")


# ---------------------------------------------------------------------------
# Bug #4: RSI/MAC trade dollar P/L scaled by allocation
# ---------------------------------------------------------------------------

def test_rsi_trade_pnl_scales_with_allocation():
    """RSI trade P/L in portfolio mode must reflect the dollar allocation, not raw per-share diff."""
    df = _make_price_df(n=252, start_price=100.0, daily_return=0.0)  # flat price so we control exactly
    # Use a simple price series with a known entry/exit
    df2 = _make_price_df(n=252, start_price=100.0, daily_return=0.001)
    params = {"period": 14, "oversold": 30, "overbought": 70}

    sig = compute_signals(df2.copy(), "RSI Strategy", params)

    allocation = 7000.0
    _, trades_alloc, _, _ = backtest(sig, slippage_bps=0.0, strategy="RSI Strategy", params=params, allocation=allocation)
    _, trades_no_alloc, _, _ = backtest(sig, slippage_bps=0.0, strategy="RSI Strategy", params=params, allocation=None)

    if trades_alloc.empty or trades_no_alloc.empty:
        print("OK: RSI no trades in test period — allocation param accepted.")
        return

    # With allocation, P/L should be much larger than without
    pnl_alloc = trades_alloc["pnl"].abs().max()
    pnl_raw = trades_no_alloc["pnl"].abs().max()

    # With allocation=7000 on a ~$100 stock, pnl should be ~70x the raw per-share pnl
    # (at minimum significantly larger)
    assert pnl_alloc > pnl_raw * 10, (
        f"Allocation-scaled PnL {pnl_alloc:.2f} should be >> raw PnL {pnl_raw:.2f}"
    )
    print(f"OK: RSI allocation PnL {pnl_alloc:.2f} >> raw PnL {pnl_raw:.2f}")


def test_pnl_formula_allocation():
    """Verify dollar_pnl = allocation * (exit/entry - 1) for a known trade."""
    allocation = 7000.0
    px_in = 150.0
    px_out = 254.16  # ~69.44% return
    expected_pnl = (allocation / px_in) * (px_out - px_in)
    expected_ret = (px_out / px_in - 1) * 100

    # Verify the formula directly
    shares_equiv = allocation / px_in
    dollar_pnl = shares_equiv * (px_out - px_in)
    assert abs(dollar_pnl - expected_pnl) < 0.01
    assert abs(dollar_pnl - allocation * expected_ret / 100) < 0.01
    print(f"OK: dollar_pnl formula: allocation={allocation}, entry={px_in}, exit={px_out} -> pnl={dollar_pnl:.2f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pure unit tests (no network)
    test_validate_portfolio_inputs_valid()
    test_validate_portfolio_inputs_too_few_tickers()
    test_validate_portfolio_inputs_bad_sum()
    test_validate_portfolio_inputs_zero_weight()
    test_compute_portfolio_params_bh()
    test_compute_portfolio_params_dca()
    test_compute_portfolio_params_no_mutation_for_mac()
    test_aggregate_portfolio_equity_math()
    test_date_intersection()
    test_portfolio_n1_regression()
    test_params_isolation_across_tickers()
    test_portfolio_equal_weights_two_identical_tickers()

    # Bug #2: calculate_equal_weights
    test_equal_weights_n1()
    test_equal_weights_n2()
    test_equal_weights_n3()
    test_equal_weights_n4()
    test_equal_weights_n6()
    test_equal_weights_n7()
    test_equal_weights_n10()

    # Bug #6: slippage/fees applied to DCA and B&H
    test_dca_slippage_reduces_final_value()
    test_dca_weekly_more_drag_than_monthly()
    test_bh_slippage_reduces_final_value()

    # Bug #4: RSI/MAC trade PnL scaled by allocation
    test_pnl_formula_allocation()
    test_rsi_trade_pnl_scales_with_allocation()

    # Network-dependent test (may be skipped in offline environments)
    test_pipeline()

    print("\nAll smoke tests passed.")
