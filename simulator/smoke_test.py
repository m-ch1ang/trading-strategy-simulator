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
    bt, trades = backtest(sig)
    assert not bt.empty
    assert {"equity", "bh_equity"}.issubset(bt.columns)
    print("OK: pipeline runs.")


if __name__ == "__main__":
    test_pipeline()
