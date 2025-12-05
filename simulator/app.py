import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i18n.i18n import t, set_language, get_lang

# -------------------------
# Helpers & Backtest Engine
# -------------------------

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str, _version: str = "v4_no_synthetic") -> tuple[pd.DataFrame, str]:
    """Load OHLCV data using Yahoo Finance as primary source and Stooq as fallback.

    Returns (df, source), where source in {"yahoo", "stooq", "error"}.
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

    # Primary: Yahoo Finance via yfinance
    try:
        import yfinance as yf
        yf_ticker = yf.Ticker(ticker.upper())
        yahoo_df = yf_ticker.history(start=start_dt, end=end_dt)
        if not yahoo_df.empty and "Close" in yahoo_df.columns:
            df = yahoo_df.rename(columns={
                "Open": "open", 
                "High": "high", 
                "Low": "low", 
                "Close": "close", 
                "Volume": "volume"
            })
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df, "yahoo"
    except Exception:
        pass

    # Fallback: Stooq via pandas-datareader
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

    # If everything failed, return empty with error
    return pd.DataFrame(), "error"


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
    elif strategy == "New Car":
        # APR-based amortized payment schedule: down payment + N loan payments
        car_price = float(params.get("car_price", 30000))
        dp = float(params.get("down_payment_amount", 0))
        term_months = int(params.get("term_months", 36))
        payment_frequency = params.get("payment_frequency", "Monthly")
        periodic_payment = float(params.get("_computed_periodic_payment", 0))
        num_loan_payments = int(params.get("_computed_num_loan_payments", term_months))

        financed = max(car_price - dp, 0)

        # Map frequency to approximate trading-day spacing (rough conversion)
        freq_days_map = {"Weekly": 5, "Biweekly": 10, "Monthly": 22}
        step_days = freq_days_map.get(payment_frequency, 22)

        df["signal"] = 0.0
        df["position_change"] = 0.0
        df["car_amount"] = 0.0  # track invested dollars each event

        # Invest down payment on first trading day
        loan_payments_made = 0
        if not df.empty and dp > 0:
            first_date = df.index[0]
            df.loc[first_date, "position_change"] = 1  # indicator for purchase
            df.loc[first_date, "car_amount"] = dp

        # Schedule N loan payments (separate from down payment)
        payment_dates = []
        if not df.empty and periodic_payment > 0 and num_loan_payments > 0:
            current_idx = 0
            while loan_payments_made < num_loan_payments:
                current_idx += step_days
                if current_idx >= len(df):
                    break
                payment_date = df.index[current_idx]
                payment_dates.append(payment_date)
                df.loc[payment_date, "position_change"] = 1
                df.loc[payment_date, "car_amount"] = periodic_payment
                loan_payments_made += 1

        # Build cumulative shares from invested capital
        cumulative_shares = 0.0
        cumulative_invested = 0.0
        for i, (date, row) in enumerate(df.iterrows()):
            invest_amt = row["car_amount"]
            if invest_amt > 0 and row["close"] > 0:
                shares = invest_amt / row["close"]
                cumulative_shares += shares
                cumulative_invested += invest_amt
            df.iloc[i, df.columns.get_loc("signal")] = cumulative_shares

        # Store summary info in params pass-through for later metrics (mutate params)
        # Total expected payments = 1 down payment (if any) + num_loan_payments
        total_expected = (1 if dp > 0 else 0) + num_loan_payments
        payments_completed = (1 if dp > 0 else 0) + loan_payments_made
        
        params["_car_total_invested"] = cumulative_invested
        params["_car_down_payment"] = dp
        params["_car_periodic_payment"] = periodic_payment
        params["_car_loan_payments_made"] = loan_payments_made
        params["_car_num_loan_payments"] = num_loan_payments
        params["_car_payments_completed"] = payments_completed
        params["_car_total_expected"] = total_expected
        params["_car_completed"] = payments_completed >= total_expected
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
    elif strategy == "New Car":
        # Similar handling to DCA but using params metadata populated earlier
        total_invested = float(params.get("_car_total_invested", 0))
        total_shares = df["signal"].iloc[-1] if not df.empty else 0.0
        if total_invested > 0 and total_shares > 0:
            portfolio_values = df["signal"] * prices
            equity = portfolio_values / total_invested
            strategy_ret = equity.pct_change().fillna(0)
        else:
            equity = pd.Series(1.0, index=df.index)
            strategy_ret = equity.pct_change().fillna(0)
        position = df["signal"]
        final_value = total_shares * prices.iloc[-1]
        unrealized_gain = final_value - total_invested
        car_metrics = {
            "total_invested": total_invested,
            "total_value": final_value,
            "total_gain": unrealized_gain,
            "total_shares": total_shares,
            "down_payment": params.get("_car_down_payment", 0),
            "periodic_payment": params.get("_car_periodic_payment", 0),
            "loan_payments_made": params.get("_car_loan_payments_made", 0),
            "num_loan_payments": params.get("_car_num_loan_payments", 0),
            "payments_completed": params.get("_car_payments_completed", 0),
            "total_expected": params.get("_car_total_expected", 0),
            "completed": params.get("_car_completed", False)
        }
        dca_metrics = car_metrics  # reuse dca_metrics dict channel for UI (or we could create new)
    else:
        # Original logic for other strategies
        position = df["signal"].shift(1).fillna(0)  # trade at next open/close assumption
        ret = prices.pct_change().fillna(0)
        strategy_ret = position * ret

        # apply simple slippage/fees on position changes
        trade_cost = abs(df["position_change"]) * (slippage_bps / 10000.0)
        strategy_ret = strategy_ret - trade_cost

        equity = (1 + strategy_ret).cumprod()

    # Buy & hold baseline or SPY strategy comparison
    ret = prices.pct_change().fillna(0)
    bh_equity = (1 + ret).cumprod()
    
    # For Buy & Hold and Dollar Cost Averaging, compare to same strategy on SPY
    if strategy in ["Buy & Hold", "Dollar Cost Averaging"]:
        try:
            # Extract date range from df
            start_date_str = df.index[0].strftime("%Y-%m-%d")
            end_date_str = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Load SPY data
            spy_df, spy_source = load_data("SPY", start_date_str, end_date_str)
            
            if not spy_df.empty and "close" in spy_df.columns:
                # Align dates: use intersection of dates
                common_dates = df.index.intersection(spy_df.index)
                
                if len(common_dates) > 0:
                    # Filter both dataframes to common dates
                    spy_df_aligned = spy_df.loc[common_dates].copy()
                    df_aligned = df.loc[common_dates].copy()
                    
                    # Apply the same strategy to SPY
                    spy_sig_df = compute_signals(spy_df_aligned, strategy, params)
                    
                    # Calculate SPY strategy equity using the same logic
                    spy_prices = spy_sig_df["close"]
                    
                    if strategy == "Dollar Cost Averaging":
                        amount = float(params.get("amount", 1000))
                        total_invested = 0
                        total_shares = 0
                        spy_equity_values = []
                        
                        for i, (date, row) in enumerate(spy_sig_df.iterrows()):
                            if row["position_change"] > 0:  # Purchase day
                                shares_bought = amount / row["close"]
                                total_shares += shares_bought
                                total_invested += amount
                            
                            # Current portfolio value
                            current_value = total_shares * row["close"]
                            spy_equity_values.append(current_value / total_invested if total_invested > 0 else 1.0)
                        
                        spy_equity = pd.Series(spy_equity_values, index=spy_sig_df.index)
                        
                        # Reindex to match original df index for comparison
                        # Forward fill and back fill to handle missing dates
                        spy_equity_aligned = spy_equity.reindex(df.index).ffill().bfill().fillna(1.0)
                        bh_equity = spy_equity_aligned
                        
                    elif strategy == "Buy & Hold":
                        amount = float(params.get("amount", 10000))
                        initial_price = spy_prices.iloc[0]
                        shares_bought = amount / initial_price
                        
                        # Calculate SPY Buy & Hold equity
                        spy_portfolio_values = shares_bought * spy_prices
                        spy_equity = spy_portfolio_values / amount  # Normalize to show growth from $1
                        
                        # Reindex to match original df index for comparison
                        spy_equity_aligned = spy_equity.reindex(df.index).ffill().bfill().fillna(1.0)
                        bh_equity = spy_equity_aligned
        except Exception:
            # If SPY loading fails, fall back to original buy & hold comparison
            pass

    # Car depreciation baseline (for New Car strategy)
    car_depreciation_equity = None
    if strategy == "New Car":
        car_price = float(params.get("car_price", 30000))
        if car_price > 0:
            # Calculate years elapsed from start date
            start_date = df.index[0]
            years_elapsed = (df.index - start_date).days / 365.25
            
            # Standard car depreciation: 20% first year, 15% per year after
            # Exponential decay model: value(t) = initial_value * (1 - rate)^years
            car_values = []
            for years in years_elapsed:
                if years <= 1.0:
                    # First year: 20% depreciation
                    remaining_value = (1.0 - 0.20) ** years
                else:
                    # After first year: 20% first year, then 15% per year
                    remaining_value = (1.0 - 0.20) * ((1.0 - 0.15) ** (years - 1.0))
                car_values.append(remaining_value)
            
            # Normalize to start at 1.0 for fair comparison with strategy equity
            car_depreciation_equity = pd.Series(car_values, index=df.index)
        else:
            car_depreciation_equity = pd.Series(1.0, index=df.index)

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
    if car_depreciation_equity is not None:
        bt["car_depreciation_equity"] = safe_column_assign(car_depreciation_equity, "car_depreciation_equity")

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
    elif strategy == "New Car":
        trades = []
        for date in df.index[df["position_change"] > 0.5]:
            px_in = prices.loc[date]
            invest_amt = df.loc[date, "car_amount"] if "car_amount" in df.columns else 0
            shares_bought = invest_amt / px_in if px_in > 0 else 0
            px_out = prices.iloc[-1]
            pnl = (px_out - px_in) * shares_bought
            ret_pct = (px_out / px_in - 1) * 100
            trades.append({"date_in": date, "date_out": df.index[-1], "pnl": pnl, "return_pct": ret_pct})
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
    st.set_page_config(page_title=t("app.title"), layout="wide")

    st.title(t("app.title"))
    st.caption(t("app.disclaimer"))

    # Instructions
    with st.expander(t("instructions.title"), expanded=False):
        st.markdown(f"""
        ### {t("instructions.getting_started")}
        1. **{t("instructions.step1")}**
        2. **{t("instructions.step2")}**
        3. **{t("instructions.step3")}**
        4. **{t("instructions.step4")}**
        5. **{t("instructions.step5")}**
        
        ### {t("instructions.ticker_guidelines")}
        - **{t("instructions.us_stocks")}**
        - **{t("instructions.non_us_stocks")}**
          - {t("instructions.uk_stocks")}
          - {t("instructions.ca_stocks")}
          - {t("instructions.de_stocks")}
          - {t("instructions.au_stocks")}
          - {t("instructions.cn_stocks")}
          - {t("instructions.hk_stocks")}
          - {t("instructions.tw_stocks")}
          - {t("instructions.sg_stocks")}
          - {t("instructions.jp_stocks")}
        - **{t("instructions.crypto")}**
        - **{t("instructions.not_sure")}**
        
        ### {t("instructions.understanding_results")}
        - **{t("instructions.price_chart")}**
        - **{t("instructions.equity_curve")}**
        - **{t("instructions.metrics")}**
        - **{t("instructions.trades_table")}**
        """)

    # Sidebar inputs
    with st.sidebar:
        # Language selector
        label_to_code = {
            "English": "en",
            "繁體中文": "zh-TW",
            "简体中文": "zh-CN",
        }
        code_to_label = {v: k for k, v in label_to_code.items()}
        
        current = get_lang()
        sel = st.selectbox(
            t("language.label") + " / 語言 / 语言",
            list(label_to_code.keys()),
            index=list(label_to_code.values()).index(current) if current in label_to_code.values() else 0
        )
        sel_code = label_to_code[sel]
        if sel_code != current:
            set_language(sel_code)
            st.rerun()
        
        st.header(t("sidebar.header"))
        ticker = st.text_input(t("sidebar.ticker_label"), value="AAPL")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(t("sidebar.start_date"), value=datetime.today() - timedelta(days=365 * 3), min_value=datetime(1950, 1, 1))
        with col2:
            end_date = st.date_input(t("sidebar.end_date"), value=datetime.today(), max_value=datetime.today())

        strategy = st.selectbox(
            t("sidebar.strategy"),
            [t("strategies.dca"), t("strategies.buy_hold"), t("strategies.ma_crossover"), t("strategies.rsi"), t("strategies.new_car")]
        )

        params = {}
        if strategy == t("strategies.ma_crossover"):
            c1, c2 = st.columns(2)
            with c1:
                params["short"] = st.number_input(t("params.short_ma"), min_value=1, max_value=250, value=20, step=1)
            with c2:
                params["long"] = st.number_input(t("params.long_ma"), min_value=2, max_value=400, value=50, step=1)
        elif strategy == t("strategies.rsi"):
            c1, c2, c3 = st.columns(3)
            with c1:
                params["period"] = st.number_input(t("params.rsi_period"), min_value=2, max_value=50, value=14, step=1)
            with c2:
                params["oversold"] = st.number_input(t("params.oversold"), min_value=1, max_value=50, value=30, step=1)
            with c3:
                params["overbought"] = st.number_input(t("params.overbought"), min_value=50, max_value=99, value=70, step=1)
        elif strategy == t("strategies.buy_hold"):
            params["amount"] = st.number_input(t("params.initial_investment"), min_value=100, value=10000, step=100)
            st.info(t("info.buy_hold_desc"))
        elif strategy == t("strategies.dca"):
            c1, c2 = st.columns(2)
            with c1:
                params["frequency"] = st.selectbox(t("params.buy_frequency"), ["Weekly", "Monthly", "Quarterly"], index=1)
            with c2:
                params["amount"] = st.number_input(t("params.dollar_amount"), min_value=100, value=1000, step=100)
        elif strategy == t("strategies.new_car"):
            st.info(t("info.new_car_desc"))
            c1, c2, c3 = st.columns(3)
            with c1:
                params["car_price"] = st.number_input(t("params.car_price"), min_value=5000, value=30000, step=500)
                params["down_payment_amount"] = st.number_input(t("params.down_payment"), min_value=0, value=6000, step=500)
            with c2:
                params["term_months"] = st.selectbox(t("params.term_months"), [12, 24, 36, 48, 60], index=2)
                params["payment_frequency"] = st.selectbox(t("params.payment_frequency"), ["Monthly", "Biweekly", "Weekly"], index=0)
            with c3:
                params["apr"] = st.number_input(t("params.apr"), min_value=0.0, max_value=25.0, value=5.0, step=0.1)

            car_price = float(params["car_price"])
            dp = float(params["down_payment_amount"])
            financed = max(car_price - dp, 0)
            apr = float(params["apr"]) / 100.0
            term_months = int(params["term_months"])
            freq = params["payment_frequency"]
            # Calculate number of loan payments (not including down payment)
            if freq == "Monthly":
                num_loan_payments = term_months
                rate_per_period = apr / 12
            elif freq == "Biweekly":
                num_loan_payments = int(round(term_months * 26 / 12))  # ~26 biweekly periods per year
                rate_per_period = apr / 26
            else:  # Weekly
                num_loan_payments = int(round(term_months * 52 / 12))  # ~52 weeks per year
                rate_per_period = apr / 52
            num_loan_payments = max(num_loan_payments, 1)
            
            # Calculate amortized payment for the financed amount only
            if rate_per_period > 0 and financed > 0:
                periodic_payment = financed * (rate_per_period * (1 + rate_per_period) ** num_loan_payments) / ((1 + rate_per_period) ** num_loan_payments - 1)
            else:
                periodic_payment = financed / num_loan_payments if num_loan_payments > 0 else 0
            
            params["_computed_periodic_payment"] = periodic_payment
            params["_computed_num_loan_payments"] = num_loan_payments
            params["_total_to_invest"] = dp + (periodic_payment * num_loan_payments)
            st.caption(t("info.car_payment_summary", dp=f"{dp:,.0f}", num_payments=num_loan_payments, payment=f"{periodic_payment:,.2f}", apr=f"{apr*100:.2f}", total=f"{dp + (periodic_payment * num_loan_payments):,.0f}"))

        slippage_bps = st.slider(t("sidebar.slippage_label"), min_value=0, max_value=50, value=0)
        run = st.button(t("sidebar.run_button"), type="primary")

    if run:
        with st.spinner(t("ui.loading")):
            try:
                df, data_source = load_data(ticker.strip().upper(), start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
                
                if df.empty:
                    if data_source == "error":
                        st.error(t("errors.fetch_error", ticker=ticker.upper()))
                        st.info(t("errors.verify_ticker"))
                        st.markdown("**[finance.yahoo.com](https://finance.yahoo.com)**")
                        st.caption(t("errors.tip"))
                        st.caption(t("errors.report_issue"))
                    else:
                        st.warning(t("errors.no_data", ticker=ticker.upper()))
                    return
                    
                # Debug info
                st.sidebar.info(t("data_source.info", source=data_source, rows=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date())))
                
                # Map translated strategy names back to internal names for compute_signals
                strategy_map = {
                    t("strategies.dca"): "Dollar Cost Averaging",
                    t("strategies.buy_hold"): "Buy & Hold",
                    t("strategies.ma_crossover"): "Moving Average Crossover",
                    t("strategies.rsi"): "RSI Strategy",
                    t("strategies.new_car"): "New Car"
                }
                internal_strategy = strategy_map.get(strategy, strategy)
                
                sig_df = compute_signals(df, internal_strategy, params)
                bt, trades_df, dca_metrics, bh_metrics = backtest(sig_df, slippage_bps=slippage_bps, strategy=internal_strategy, params=params)
                
            except Exception as e:
                st.error(t("errors.backtest_error", error=str(e)))
                st.error(t("errors.try_again"))
                return

        # Data source banner
        if data_source == "yahoo":
            st.success(t("data_source.yahoo"))
        elif data_source == "stooq":
            st.success(t("data_source.stooq"))

        # Price chart with markers
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=bt.index, y=bt["price"], mode="lines", name=t("charts.price")))
        buys = sig_df.index[sig_df["position_change"] > 0.5]
        sells = sig_df.index[sig_df["position_change"] < -0.5]
        price_fig.add_trace(go.Scatter(x=buys, y=sig_df.loc[buys, "close"], mode="markers", name=t("charts.buy"), marker=dict(color="green", symbol="triangle-up", size=10)))
        price_fig.add_trace(go.Scatter(x=sells, y=sig_df.loc[sells, "close"], mode="markers", name=t("charts.sell"), marker=dict(color="red", symbol="triangle-down", size=10)))
        price_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))

        # Equity curves
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], mode="lines", name=t("charts.strategy")))
        # Use car depreciation for New Car strategy, SPY strategy for Buy & Hold/DCA, buy & hold for others
        if internal_strategy == "New Car" and "car_depreciation_equity" in bt.columns:
            eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["car_depreciation_equity"], mode="lines", name=t("charts.car_depreciation")))
        elif internal_strategy == "Buy & Hold":
            spy_bh_label = t("charts.spy_buy_hold")
            if spy_bh_label == "charts.spy_buy_hold":
                spy_bh_label = "SPY Buy & Hold"
            eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], mode="lines", name=spy_bh_label))
        elif internal_strategy == "Dollar Cost Averaging":
            spy_dca_label = t("charts.spy_dca")
            if spy_dca_label == "charts.spy_dca":
                spy_dca_label = "SPY Dollar Cost Averaging"
            eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], mode="lines", name=spy_dca_label))
        else:
            eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["bh_equity"], mode="lines", name=t("charts.buy_hold")))
        eq_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))

        # Metrics - ensure we get scalar values, not Series
        total_return = float(bt["equity"].iloc[-1]) - 1
        sr = sharpe_ratio(bt["strategy_ret"])
        mdd = max_drawdown(bt["equity"])  # negative number
        
        # Calculate comparison return (car depreciation for New Car, SPY strategy for Buy & Hold/DCA, buy & hold for others)
        if internal_strategy == "New Car" and "car_depreciation_equity" in bt.columns:
            comparison_return = float(bt["car_depreciation_equity"].iloc[-1]) - 1
            comparison_label = t("metrics.car_depreciation_return")
        elif internal_strategy == "Buy & Hold":
            comparison_return = float(bt["bh_equity"].iloc[-1]) - 1
            comparison_label = t("metrics.spy_buy_hold_return")
            if comparison_label == "metrics.spy_buy_hold_return":
                comparison_label = "SPY Buy & Hold Return"
        elif internal_strategy == "Dollar Cost Averaging":
            comparison_return = float(bt["bh_equity"].iloc[-1]) - 1
            comparison_label = t("metrics.spy_dca_return")
            if comparison_label == "metrics.spy_dca_return":
                comparison_label = "SPY Dollar Cost Averaging Return"
        else:
            comparison_return = float(bt["bh_equity"].iloc[-1]) - 1
            comparison_label = t("metrics.buy_hold_return")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("metrics.total_return"), f"{total_return*100:.2f}%")
        c2.metric(t("metrics.sharpe_ratio"), f"{sr:.2f}")
        c3.metric(t("metrics.max_drawdown"), f"{mdd*100:.2f}%")
        c4.metric(comparison_label, f"{comparison_return*100:.2f}%")

        # DCA-specific metrics
        if internal_strategy == "Dollar Cost Averaging" and dca_metrics and "down_payment" not in dca_metrics:
            st.subheader(t("dca_metrics.title"))
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric(t("dca_metrics.total_invested"), f"${dca_metrics['total_invested']:,.2f}")
            dc2.metric(t("dca_metrics.total_value"), f"${dca_metrics['total_value']:,.2f}")
            dc3.metric(t("dca_metrics.total_gain"), f"${dca_metrics['total_gain']:,.2f}", 
                      delta=f"{(dca_metrics['total_gain']/dca_metrics['total_invested']*100):+.2f}%")
            
            st.info(t("dca_metrics.total_shares", shares=f"{dca_metrics['total_shares']:.4f}"))
        if internal_strategy == "New Car" and dca_metrics and "down_payment" in dca_metrics:
            st.subheader(t("new_car_metrics.title"))
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric(t("new_car_metrics.total_invested"), f"${dca_metrics['total_invested']:,.2f}")
            nc2.metric(t("new_car_metrics.total_value"), f"${dca_metrics['total_value']:,.2f}")
            pct = (dca_metrics['total_gain']/dca_metrics['total_invested']*100) if dca_metrics['total_invested']>0 else 0
            nc3.metric(t("new_car_metrics.total_gain"), f"${dca_metrics['total_gain']:,.2f}", delta=f"{pct:+.2f}%")
            
            # Display payment progress with correct tracking
            loan_pmts = dca_metrics.get('loan_payments_made', 0)
            total_loan = dca_metrics.get('num_loan_payments', 0)
            payments_completed = dca_metrics.get('payments_completed', 0)
            total_expected = dca_metrics.get('total_expected', 0)
            status_text = t("new_car_metrics.completed") if dca_metrics.get('completed') else t("new_car_metrics.in_progress")
            
            apr_display = params.get('apr', None)
            apr_text = f" | APR: {apr_display:.2f}%" if apr_display is not None else ""
            
            st.info(t("new_car_metrics.payment_progress", completed=payments_completed, total=total_expected, status=status_text))
            st.caption(t("new_car_metrics.payment_details", dp=f"{dca_metrics['down_payment']:,.2f}", loan_pmts=loan_pmts, total_loan=total_loan, payment=f"{dca_metrics['periodic_payment']:,.2f}", apr=apr_text, shares=f"{dca_metrics['total_shares']:.4f}"))

        # Buy & Hold specific metrics
        if internal_strategy == "Buy & Hold" and bh_metrics:
            st.subheader(t("buy_hold_metrics.title"))
            bh1, bh2, bh3 = st.columns(3)
            bh1.metric(t("buy_hold_metrics.total_invested"), f"${bh_metrics['total_invested']:,.2f}")
            bh2.metric(t("buy_hold_metrics.total_value"), f"${bh_metrics['total_value']:,.2f}")
            bh3.metric(t("buy_hold_metrics.total_gain"), f"${bh_metrics['total_gain']:,.2f}", 
                      delta=f"{(bh_metrics['total_gain']/bh_metrics['total_invested']*100):+.2f}%")
            
            st.info(t("buy_hold_metrics.total_shares", shares=f"{bh_metrics['total_shares']:.4f}"))

        # Layout charts
        st.subheader(t("charts.price_title", ticker=ticker.upper()))
        st.plotly_chart(price_fig, use_container_width=True)

        # Update equity title based on comparison type; fallback to English literal if translation missing
        if internal_strategy in ["Buy & Hold", "Dollar Cost Averaging"]:
            equity_title = t("charts.equity_title_spy")
            if equity_title == "charts.equity_title_spy":
                equity_title = "Equity Curve: Strategy vs S&P 500"
        else:
            equity_title = t("charts.equity_title")
        st.subheader(equity_title)
        st.plotly_chart(eq_fig, use_container_width=True)

        # Trades table
        st.subheader(t("trades.title"))
        if not trades_df.empty:
            tshow = trades_df.copy()
            tshow["date_in"] = pd.to_datetime(tshow["date_in"]).dt.strftime("%Y-%m-%d")
            tshow["date_out"] = pd.to_datetime(tshow["date_out"]).dt.strftime("%Y-%m-%d")
            tshow["pnl"] = tshow["pnl"].map(lambda x: f"${float(x):,.2f}")
            tshow["return_pct"] = tshow["return_pct"].map(lambda x: f"{float(x):.2f}%")
            # Rename columns for readability
            tshow = tshow.rename(columns={
                "date_in": t("trades.entry_date"),
                "date_out": t("trades.exit_date"),
                "pnl": t("trades.pnl"),
                "return_pct": t("trades.return_pct")
            })
            st.dataframe(tshow, use_container_width=True, hide_index=True)
        else:
            st.info(t("trades.no_trades"))

    else:
        st.info(t("ui.waiting"))


if __name__ == "__main__":
    main()
