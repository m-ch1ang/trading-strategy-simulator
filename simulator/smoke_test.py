import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app import (
    load_data, compute_signals, backtest, sharpe_ratio, max_drawdown,
    validate_portfolio_inputs, compute_portfolio_params,
    aggregate_portfolio_equity, run_portfolio_backtest, PortfolioResult,
)


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
    assert abs(result["amount"] - 2000.0) < 1e-6
    assert base["amount"] == 500, "base_params must not be mutated"
    print("OK: compute_portfolio_params scales amount for DCA.")


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

    assert abs(p1["amount"] - 6000.0) < 1e-6
    assert abs(p2["amount"] - 4000.0) < 1e-6
    assert base["amount"] == 1000, "base must be untouched"
    # Mutating p1 must not affect p2
    p1["amount"] = 99999
    assert p2["amount"] == 4000.0
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

    # Network-dependent test (may be skipped in offline environments)
    test_pipeline()

    print("\nAll smoke tests passed.")
