import pandas as pd
from datetime import datetime, timedelta
from app import load_data, compute_signals, backtest


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


if __name__ == "__main__":
    test_pipeline()
