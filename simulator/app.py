import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

# -------------------------
# Helpers & Backtest Engine
# -------------------------

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Load OHLCV data with multiple fallbacks.

    Returns (df, source), where source in {"yahoo", "ticker.history", "yahoo-10y", "stooq", "synthetic", "none"}.
    """
    if not ticker:
        return pd.DataFrame(), "none"

    # Normalize dates
    try:
        start_dt = pd.to_datetime(start).tz_localize(None)
    except Exception:
        start_dt = pd.Timestamp.today() - pd.Timedelta(days=365)
    try:
        end_dt = pd.to_datetime(end).tz_localize(None)
    except Exception:
        end_dt = pd.Timestamp.today()
    if start_dt >= end_dt:
        end_dt = start_dt + pd.Timedelta(days=1)

    # Try direct download first
    last_err = None
    for _ in range(2):
        try:
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                return df, "yahoo"
        except Exception as e:
            last_err = e

    # Fallback: Ticker.history with explicit start/end
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(start=start_dt, end=end_dt, interval="1d", auto_adjust=True)
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
            df = hist.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df, "ticker.history"
    except Exception as e:
        last_err = e

    # Final Fallback: pull a longer period and slice locally
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="10y", interval="1d", auto_adjust=True)
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            df = hist.loc[(hist.index >= start_dt) & (hist.index <= end_dt)].copy()
            if not df.empty:
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df.index.name = "date"
                return df, "yahoo-10y"
    except Exception as e:
        last_err = e

    # Stooq fallback via pandas-datareader (note: limited symbols)
    try:
        stq = pdr.DataReader(ticker, "stooq", start_dt, end_dt)
        if isinstance(stq, pd.DataFrame) and not stq.empty and "Close" in stq.columns:
            stq = stq.sort_index()
            df = stq.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df, "stooq"
    except Exception:
        pass

    # Synthetic fallback to keep UI functional
    try:
        idx = pd.date_range(start_dt, end_dt, freq="B")
        if len(idx) >= 20:
            np.random.seed(42)
            steps = np.random.normal(0, 0.01, size=len(idx))
            price = 100 * np.exp(np.cumsum(steps))
            df = pd.DataFrame({
                "open": price * (1 + np.random.normal(0, 0.001, size=len(idx))),
                "high": price * (1 + abs(np.random.normal(0, 0.002, size=len(idx)))),
                "low": price * (1 - abs(np.random.normal(0, 0.002, size=len(idx)))),
                "close": price,
                "volume": np.random.randint(1e5, 5e5, size=len(idx)),
            }, index=idx)
            df.index.name = "date"
            return df, "synthetic"
    except Exception:
        pass

    # If everything failed, return empty
    return pd.DataFrame(), "none"


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    df = df.copy()
    if strategy == "Moving Average Crossover":
        short = int(params.get("short", 20))
        long = int(params.get("long", 50))
        if short >= long:
            long = short + 1
        df["ma_short"] = df["close"].rolling(short).mean()
        df["ma_long"] = df["close"].rolling(long).mean()
        df["signal"] = 0
        df.loc[df.index[long-1]:, "signal"] = np.where(df.loc[df.index[long-1]:, "ma_short"] > df.loc[df.index[long-1]:, "ma_long"], 1, 0)
        df["position_change"] = df["signal"].diff().fillna(0)
    elif strategy == "RSI Strategy":
        period = int(params.get("period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        df["rsi"] = calc_rsi(df["close"], period)
        df["signal"] = 0
        # Enter long when RSI crosses up oversold; exit when crosses down overbought
        df["long_entry"] = (df["rsi"].shift(1) < oversold) & (df["rsi"] >= oversold)
        df["long_exit"] = (df["rsi"].shift(1) > overbought) & (df["rsi"] <= overbought)
        df.loc[df["long_entry"], "signal"] = 1
        df.loc[df["long_exit"], "signal"] = 0
        df["signal"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)
        df["position_change"] = df["signal"].diff().fillna(0)
    else:
        df["signal"] = 0
        df["position_change"] = 0
    return df


def backtest(df: pd.DataFrame, slippage_bps: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame(columns=["date_in", "date_out", "pnl", "return_pct"]) 

    prices = df["close"]
    position = df["signal"].shift(1).fillna(0)  # trade at next open/close assumption

    ret = prices.pct_change().fillna(0)
    strategy_ret = position * ret

    # apply simple slippage/fees on position changes
    trade_cost = abs(df["position_change"]) * (slippage_bps / 10000.0)
    strategy_ret = strategy_ret - trade_cost

    equity = (1 + strategy_ret).cumprod()

    # Buy & hold baseline
    bh_equity = (1 + ret).cumprod()

    # Create DataFrame using a safe, conservative approach
    bt = pd.DataFrame(index=df.index)
    
    def safe_column_assign(data, col_name):
        """Safely assign data to DataFrame column with debugging"""
        try:
            # If it's already a pandas Series with the right index, use it directly
            if isinstance(data, pd.Series) and len(data) == len(df.index):
                return data
            
            # If it has values attribute, extract it carefully
            if hasattr(data, 'values'):
                vals = data.values
                # Only squeeze if it's exactly (n, 1) shape
                if vals.ndim == 2 and vals.shape == (len(df.index), 1):
                    vals = vals.squeeze()
                elif vals.ndim == 1 and len(vals) == len(df.index):
                    pass  # Already correct
                else:
                    # Fall back to original data if shapes don't make sense
                    return pd.Series(data, index=df.index)
                return pd.Series(vals, index=df.index)
            
            # For everything else, create a new Series
            return pd.Series(data, index=df.index)
            
        except Exception:
            # Last resort: force it to work by taking only the right number of elements
            arr = np.array(data).flatten()
            if len(arr) >= len(df.index):
                return pd.Series(arr[:len(df.index)], index=df.index)
            else:
                raise ValueError(f"Cannot create series for {col_name}: insufficient data")
    
    # Assign columns safely
    bt["price"] = safe_column_assign(prices, "price")
    bt["position"] = safe_column_assign(position, "position")
    bt["ret"] = safe_column_assign(ret, "ret")
    bt["strategy_ret"] = safe_column_assign(strategy_ret, "strategy_ret")
    bt["equity"] = safe_column_assign(equity, "equity")
    bt["bh_equity"] = safe_column_assign(bh_equity, "bh_equity")

    # Extract trades from position change signals
    trade_entries = df.index[df["position_change"] > 0.5]
    trade_exits = df.index[df["position_change"] < -0.5]

    # If an entry without an exit by end, assume exit on last bar
    entries = list(trade_entries)
    exits = list(trade_exits)
    trades = []
    i = j = 0
    while i < len(entries):
        entry_date = entries[i]
        # find the first exit after entry
        exit_date = None
        while j < len(exits) and exits[j] <= entry_date:
            j += 1
        if j < len(exits):
            exit_date = exits[j]
            j += 1
        else:
            exit_date = df.index[-1]
        px_in = prices.loc[entry_date]
        px_out = prices.loc[exit_date]
        pnl = px_out - px_in
        ret_pct = (px_out / px_in - 1) * 100
        trades.append({"date_in": entry_date, "date_out": exit_date, "pnl": pnl, "return_pct": ret_pct})
        i += 1

    trades_df = pd.DataFrame(trades)
    return bt, trades_df


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    sr = (returns.mean() / returns.std(ddof=0)) * np.sqrt(periods_per_year)
    return float(sr)


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return float(drawdown.min())


# -------------------------
# UI
# -------------------------

def main():
    st.set_page_config(page_title="Trading Strategy Simulator", layout="wide")

    st.title("Trading Strategy Simulator")
    st.caption("DISCLAIMER: Investing in financial markets involves risk. Past performance is not indicative of future results. Investors may experience partial or total loss of capital. This tool is for educational and informational purposes only and does not constitute financial advice. Always conduct your own research or consult a qualified financial advisor before making investment decisions.")

    # Sidebar inputs
    with st.sidebar:
        st.header("Parameters")
        ticker = st.text_input("Stock Ticker", value="AAPL")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=365 * 3))
        with col2:
            end_date = st.date_input("End Date", value=datetime.today())

        strategy = st.selectbox("Strategy", ["Moving Average Crossover", "RSI Strategy"])

        params = {}
        if strategy == "Moving Average Crossover":
            c1, c2 = st.columns(2)
            with c1:
                params["short"] = st.number_input("Short MA", min_value=1, max_value=250, value=20, step=1)
            with c2:
                params["long"] = st.number_input("Long MA", min_value=2, max_value=400, value=50, step=1)
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                params["period"] = st.number_input("RSI Period", min_value=2, max_value=50, value=14, step=1)
            with c2:
                params["oversold"] = st.number_input("Oversold", min_value=1, max_value=50, value=30, step=1)
            with c3:
                params["overbought"] = st.number_input("Overbought", min_value=50, max_value=99, value=70, step=1)

        slippage_bps = st.slider("Slippage/Fees (bps per trade)", min_value=0, max_value=50, value=0)
        run = st.button("Run Strategy", type="primary")

    if run:
        with st.spinner("Loading data and running backtest..."):
            df, data_source = load_data(ticker.strip().upper(), start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
            if df.empty:
                st.warning("No data loaded. Check ticker or date range.")
                return
            sig_df = compute_signals(df, strategy, params)
            bt, trades_df = backtest(sig_df, slippage_bps=slippage_bps)

        # Data source banner
        if data_source in {"stooq", "synthetic"}:
            if data_source == "synthetic":
                st.info("Using synthetic data because live data could not be fetched. Charts are for demo only.")
            else:
                st.info("Using Stooq data fallback. Live data may differ from Yahoo Finance.")

        # Price chart with markers
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=bt.index, y=bt["price"], mode="lines", name="Price"))
        buys = sig_df.index[sig_df["position_change"] > 0.5]
        sells = sig_df.index[sig_df["position_change"] < -0.5]
        price_fig.add_trace(go.Scatter(x=buys, y=sig_df.loc[buys, "close"], mode="markers", name="Buy", marker=dict(color="green", symbol="triangle-up", size=10)))
        price_fig.add_trace(go.Scatter(x=sells, y=sig_df.loc[sells, "close"], mode="markers", name="Sell", marker=dict(color="red", symbol="triangle-down", size=10)))
        price_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))

        # Equity curves
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], mode="lines", name="Strategy"))
        eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], mode="lines", name="Buy & Hold"))
        eq_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))

        # Metrics - ensure we get scalar values, not Series
        total_return = float(bt["equity"].iloc[-1]) - 1
        bh_return = float(bt["bh_equity"].iloc[-1]) - 1
        sr = sharpe_ratio(bt["strategy_ret"])
        mdd = max_drawdown(bt["equity"])  # negative number

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{total_return*100:.2f}%")
        c2.metric("Sharpe Ratio", f"{sr:.2f}")
        c3.metric("Max Drawdown", f"{mdd*100:.2f}%")
        c4.metric("Buy & Hold Return", f"{bh_return*100:.2f}%")

        # Layout charts
        st.subheader(f"{ticker.upper()} Price and Signals")
        st.plotly_chart(price_fig, use_container_width=True)

        st.subheader("Equity Curve: Strategy vs Buy & Hold")
        st.plotly_chart(eq_fig, use_container_width=True)

        # Trades table
        st.subheader("Trades")
        if not trades_df.empty:
            tshow = trades_df.copy()
            tshow["date_in"] = pd.to_datetime(tshow["date_in"]).dt.strftime("%Y-%m-%d")
            tshow["date_out"] = pd.to_datetime(tshow["date_out"]).dt.strftime("%Y-%m-%d")
            tshow["pnl"] = tshow["pnl"].map(lambda x: f"${float(x):,.2f}")
            tshow["return_pct"] = tshow["return_pct"].map(lambda x: f"{float(x):.2f}%")
            st.dataframe(tshow, use_container_width=True, hide_index=True)
        else:
            st.info("No completed trades in the period.")

    else:
        st.info("Set parameters and click 'Run Strategy' in the sidebar.")


if __name__ == "__main__":
    main()
