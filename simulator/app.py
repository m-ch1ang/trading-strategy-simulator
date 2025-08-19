import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

# -------------------------
# Helpers & Backtest Engine
# -------------------------

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str, _version: str = "v2_stooq_only") -> tuple[pd.DataFrame, str]:
    """Load OHLCV data using Stooq as primary source with synthetic fallback.

    Returns (df, source), where source in {"stooq", "synthetic", "none"}.
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

    # Primary: Stooq via pandas-datareader
    try:
        # Stooq uses different ticker formats for some exchanges
        # Try the ticker as-is first, then with common suffixes
        tickers_to_try = [ticker.upper()]
        
        # Add common US market suffixes if not already present
        if '.' not in ticker and len(ticker) <= 5:
            tickers_to_try.extend([f"{ticker.upper()}.US", f"{ticker.upper()}.NASDAQ", f"{ticker.upper()}.NYSE"])
        
        for test_ticker in tickers_to_try:
            try:
                stq = pdr.DataReader(test_ticker, "stooq", start_dt, end_dt)
                if isinstance(stq, pd.DataFrame) and not stq.empty and "Close" in stq.columns:
                    stq = stq.sort_index()  # Stooq data is often in reverse chronological order
                    df = stq.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "date"
                    return df, "stooq"
            except Exception:
                continue
    except Exception:
        pass

    # Synthetic fallback to keep UI functional
    try:
        # Create a more robust date range
        start_date = pd.Timestamp(start_dt).normalize()
        end_date = pd.Timestamp(end_dt).normalize()
        
        # Generate business days between start and end
        idx = pd.date_range(start_date, end_date, freq="D")
        idx = idx[idx.dayofweek < 5]  # Remove weekends
        
        if len(idx) >= 10:  # Need at least 10 days of data
            np.random.seed(42)  # Consistent seed for reproducible data
            n_days = len(idx)
            
            # Generate realistic price movement
            initial_price = 100.0
            daily_returns = np.random.normal(0.0008, 0.02, n_days)  # ~0.2% daily drift, 2% volatility
            price_series = initial_price * np.exp(np.cumsum(daily_returns))
            
            # Generate OHLC data
            noise_factor = 0.005
            df = pd.DataFrame({
                "open": price_series * (1 + np.random.normal(0, noise_factor, n_days)),
                "high": price_series * (1 + np.abs(np.random.normal(0, noise_factor, n_days))),
                "low": price_series * (1 - np.abs(np.random.normal(0, noise_factor, n_days))),
                "close": price_series,
                "volume": np.random.randint(50000, 200000, n_days),
            }, index=idx)
            
            # Ensure high >= close >= low and high >= open >= low
            df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
            df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
            
            df.index.name = "date"
            return df, "synthetic"
    except Exception as e:
        st.error(f"Synthetic data generation failed: {str(e)}")
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
            
        # Calculate moving averages
        df["ma_short"] = df["close"].rolling(short).mean()
        df["ma_long"] = df["close"].rolling(long).mean()
        
        # Initialize signal column
        df["signal"] = 0
        
        # Only compute signals where we have valid MA values
        valid_idx = df["ma_long"].notna()
        
        # Generate signals: 1 when short MA > long MA, 0 otherwise
        df.loc[valid_idx, "signal"] = np.where(
            df.loc[valid_idx, "ma_short"] > df.loc[valid_idx, "ma_long"], 1, 0
        )
        
        # Calculate position changes (entry/exit points)
        df["position_change"] = df["signal"].diff().fillna(0)
    elif strategy == "RSI Strategy":
        period = int(params.get("period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        
        # Calculate RSI
        df["rsi"] = calc_rsi(df["close"], period)
        
        # Initialize signal
        df["signal"] = 0
        
        # Only compute where RSI is valid
        valid_idx = df["rsi"].notna()
        
        # Simple RSI strategy: long when RSI < oversold, flat when RSI > overbought
        df.loc[valid_idx & (df["rsi"] <= oversold), "signal"] = 1  # Buy signal
        df.loc[valid_idx & (df["rsi"] >= overbought), "signal"] = 0  # Sell signal
        
        # Forward fill signals to maintain positions
        df["signal"] = df["signal"].replace(0, np.nan).fillna(method="ffill").fillna(0)
        
        # Calculate position changes
        df["position_change"] = df["signal"].diff().fillna(0)
    elif strategy == "Buy & Hold":
        amount = float(params.get("amount", 10000))
        # Calculate shares bought with initial investment
        initial_price = df["close"].iloc[0]
        shares_bought = amount / initial_price
        
        # Always hold the same number of shares
        df["signal"] = shares_bought  # Number of shares held
        df["position_change"] = 0  # No position changes after initial buy
        # Set initial buy signal
        df.iloc[0, df.columns.get_loc("position_change")] = shares_bought  # Initial buy
        df["bh_amount"] = amount  # Track initial investment amount
    elif strategy == "Dollar Cost Averaging":
        frequency = params.get("frequency", "Monthly")
        amount = float(params.get("amount", 1000))
        
        # Initialize signals
        df["signal"] = 0
        df["position_change"] = 0
        df["dca_amount"] = 0.0  # Track dollar amounts
        
        # Determine frequency in business days
        freq_map = {"Weekly": 5, "Monthly": 22, "Quarterly": 66}
        freq_days = freq_map.get(frequency, 22)
        
        # Calculate DCA purchase dates
        purchase_dates = []
        current_idx = 0
        while current_idx < len(df):
            purchase_dates.append(df.index[current_idx])
            current_idx += freq_days
        
        # Mark purchase signals
        for date in purchase_dates:
            if date in df.index:
                df.loc[date, "position_change"] = 1  # Buy signal
                df.loc[date, "dca_amount"] = amount
        
        # Calculate cumulative position (shares owned)
        cumulative_shares = 0
        for i, (date, row) in enumerate(df.iterrows()):
            if row["position_change"] > 0:
                # Buy more shares with fixed dollar amount
                shares_bought = amount / row["close"]
                cumulative_shares += shares_bought
            df.iloc[i, df.columns.get_loc("signal")] = cumulative_shares
    else:
        df["signal"] = 0
        df["position_change"] = 0
    return df


def backtest(df: pd.DataFrame, slippage_bps: float = 0.0, strategy: str = "", params: dict = {}) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    if df.empty:
        return df, pd.DataFrame(columns=["date_in", "date_out", "pnl", "return_pct"]), {}, {}

    prices = df["close"]
    dca_metrics = {}
    bh_metrics = {}
    
    # Handle DCA strategy differently
    if strategy == "Dollar Cost Averaging":
        amount = float(params.get("amount", 1000))
        
        # Calculate returns based on actual dollar investments
        total_invested = 0
        total_shares = 0
        equity_values = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            if row["position_change"] > 0:  # Purchase day
                shares_bought = amount / row["close"]
                total_shares += shares_bought
                total_invested += amount
            
            # Current portfolio value
            current_value = total_shares * row["close"]
            equity_values.append(current_value / total_invested if total_invested > 0 else 1.0)
        
        # Calculate DCA-specific metrics
        final_value = total_shares * prices.iloc[-1]
        unrealized_gain = final_value - total_invested
        
        dca_metrics = {
            "total_invested": total_invested,
            "total_value": final_value,
            "total_gain": unrealized_gain,
            "total_shares": total_shares
        }
        
        equity = pd.Series(equity_values, index=df.index)
        strategy_ret = equity.pct_change().fillna(0)
        position = df["signal"]  # Use the cumulative shares as position
    elif strategy == "Buy & Hold":
        amount = float(params.get("amount", 10000))
        initial_price = prices.iloc[0]
        shares_bought = amount / initial_price
        
        # Calculate Buy & Hold metrics
        final_value = shares_bought * prices.iloc[-1]
        total_gain = final_value - amount
        
        bh_metrics = {
            "total_invested": amount,
            "total_value": final_value,
            "total_gain": total_gain,
            "total_shares": shares_bought
        }
        
        # Calculate returns based on actual dollar performance
        portfolio_values = shares_bought * prices
        equity = portfolio_values / amount  # Normalize to show growth from $1
        strategy_ret = equity.pct_change().fillna(0)
        position = pd.Series(shares_bought, index=df.index)
    else:
        # Original logic for other strategies
        position = df["signal"].shift(1).fillna(0)  # trade at next open/close assumption
        ret = prices.pct_change().fillna(0)
        strategy_ret = position * ret

        # apply simple slippage/fees on position changes
        trade_cost = abs(df["position_change"]) * (slippage_bps / 10000.0)
        strategy_ret = strategy_ret - trade_cost

        equity = (1 + strategy_ret).cumprod()

    # Buy & hold baseline
    ret = prices.pct_change().fillna(0)
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
    if strategy == "Dollar Cost Averaging":
        # For DCA, each purchase is a separate "trade"
        amount = float(params.get("amount", 1000))
        trades = []
        for date in df.index[df["position_change"] > 0.5]:
            px_in = prices.loc[date]
            shares_bought = amount / px_in
            # For DCA, we show each purchase as a trade from purchase to end
            px_out = prices.iloc[-1]  # Final price
            pnl = (px_out - px_in) * shares_bought
            ret_pct = (px_out / px_in - 1) * 100
            trades.append({"date_in": date, "date_out": df.index[-1], "pnl": pnl, "return_pct": ret_pct})
        trades_df = pd.DataFrame(trades)
    elif strategy == "Buy & Hold":
        # For Buy & Hold, show one trade from start to end
        amount = float(params.get("amount", 10000))
        px_in = prices.iloc[0]
        px_out = prices.iloc[-1]
        shares_bought = amount / px_in
        pnl = (px_out - px_in) * shares_bought
        ret_pct = (px_out / px_in - 1) * 100
        trades = [{"date_in": df.index[0], "date_out": df.index[-1], "pnl": pnl, "return_pct": ret_pct}]
        trades_df = pd.DataFrame(trades)
    else:
        # Original trade extraction logic for other strategies
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
    return bt, trades_df, dca_metrics, bh_metrics


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

        strategy = st.selectbox("Strategy", ["Buy & Hold", "Dollar Cost Averaging", "Moving Average Crossover", "RSI Strategy"])

        params = {}
        if strategy == "Moving Average Crossover":
            c1, c2 = st.columns(2)
            with c1:
                params["short"] = st.number_input("Short MA", min_value=1, max_value=250, value=20, step=1)
            with c2:
                params["long"] = st.number_input("Long MA", min_value=2, max_value=400, value=50, step=1)
        elif strategy == "RSI Strategy":
            c1, c2, c3 = st.columns(3)
            with c1:
                params["period"] = st.number_input("RSI Period", min_value=2, max_value=50, value=14, step=1)
            with c2:
                params["oversold"] = st.number_input("Oversold", min_value=1, max_value=50, value=30, step=1)
            with c3:
                params["overbought"] = st.number_input("Overbought", min_value=50, max_value=99, value=70, step=1)
        elif strategy == "Buy & Hold":
            params["amount"] = st.number_input("Initial Investment Amount", min_value=100, max_value=100000, value=10000, step=100)
            st.info("Buy & Hold strategy: Buy at the start and hold until the end.")
        elif strategy == "Dollar Cost Averaging":
            c1, c2 = st.columns(2)
            with c1:
                params["frequency"] = st.selectbox("Buy Frequency", ["Weekly", "Monthly", "Quarterly"], index=1)
            with c2:
                params["amount"] = st.number_input("Dollar Amount per Purchase", min_value=100, max_value=10000, value=1000, step=100)

        slippage_bps = st.slider("Slippage/Fees (bps per trade)", min_value=0, max_value=50, value=0)
        run = st.button("Run Strategy", type="primary")

    if run:
        with st.spinner("Loading data and running backtest..."):
            try:
                df, data_source = load_data(ticker.strip().upper(), start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
                
                if df.empty:
                    st.warning(f"No data loaded for {ticker.upper()}. Data source attempted: {data_source}. Check ticker or date range.")
                    return
                    
                # Debug info
                st.sidebar.info(f"Data source: {data_source} | Rows: {len(df)} | Date range: {df.index[0].date()} to {df.index[-1].date()}")
                
                sig_df = compute_signals(df, strategy, params)
                bt, trades_df, dca_metrics, bh_metrics = backtest(sig_df, slippage_bps=slippage_bps, strategy=strategy, params=params)
                
            except Exception as e:
                st.error(f"Error during backtesting: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")
                return

        # Data source banner
        if data_source == "synthetic":
            st.info("📊 Using synthetic data because historical data could not be fetched. Charts are for demo only.")
        elif data_source == "stooq":
            st.success("📈 Using Stooq historical data")

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

        # DCA-specific metrics
        if strategy == "Dollar Cost Averaging" and dca_metrics:
            st.subheader("Dollar Cost Averaging Metrics")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Total Invested", f"${dca_metrics['total_invested']:,.2f}")
            dc2.metric("Total Value", f"${dca_metrics['total_value']:,.2f}")
            dc3.metric("Total Gain", f"${dca_metrics['total_gain']:,.2f}", 
                      delta=f"{(dca_metrics['total_gain']/dca_metrics['total_invested']*100):+.2f}%")
            
            st.info(f"Total shares owned: {dca_metrics['total_shares']:.4f}")

        # Buy & Hold specific metrics
        if strategy == "Buy & Hold" and bh_metrics:
            st.subheader("Buy & Hold Metrics")
            bh1, bh2, bh3 = st.columns(3)
            bh1.metric("Total Invested", f"${bh_metrics['total_invested']:,.2f}")
            bh2.metric("Total Value", f"${bh_metrics['total_value']:,.2f}")
            bh3.metric("Total Gain", f"${bh_metrics['total_gain']:,.2f}", 
                      delta=f"{(bh_metrics['total_gain']/bh_metrics['total_invested']*100):+.2f}%")
            
            st.info(f"Total shares owned: {bh_metrics['total_shares']:.4f}")

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
            # Rename columns for readability
            tshow = tshow.rename(columns={
                "date_in": "Entry Date",
                "date_out": "Exit Date",
                "pnl": "Profit / Loss",
                "return_pct": "Return (%)"
            })
            st.dataframe(tshow, use_container_width=True, hide_index=True)
        else:
            st.info("No completed trades in the period.")

    else:
        st.info("Set parameters and click 'Run Strategy' in the sidebar.")


if __name__ == "__main__":
    main()
