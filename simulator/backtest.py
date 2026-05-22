import pandas as pd
import numpy as np

from data import load_data
from strategies import compute_signals


def backtest(df: pd.DataFrame, slippage_bps: float = 0.0, strategy: str = "", params: dict = {}, allocation: float = None) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    if df.empty:
        return df, pd.DataFrame(columns=["date_in", "date_out", "pnl", "return_pct"]), {}, {}

    prices = df["close"]
    dca_metrics = {}
    bh_metrics = {}

    # Handle DCA strategy differently
    if strategy == "Dollar Cost Averaging":
        amount = float(params.get("amount", 1000))
        fee_rate = slippage_bps / 10000.0

        # Calculate returns based on actual dollar investments
        total_invested = 0
        total_shares = 0
        equity_values = []

        for i, (date, row) in enumerate(df.iterrows()):
            if row["position_change"] > 0:  # Purchase day
                effective_amount = amount * (1.0 - fee_rate)
                shares_bought = effective_amount / row["close"]
                total_shares += shares_bought
                total_invested += amount  # track nominal amount paid

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
        fee_rate = slippage_bps / 10000.0
        effective_amount = amount * (1.0 - fee_rate)
        shares_bought = effective_amount / initial_price

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
        equity = portfolio_values / amount  # Normalize vs nominal invested
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
        dca_metrics = car_metrics  # reuse dca_metrics dict channel for UI
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

                    # Apply the same strategy to SPY
                    spy_sig_df = compute_signals(spy_df_aligned, strategy, params)

                    # Calculate SPY strategy equity using the same logic
                    spy_prices = spy_sig_df["close"]

                    if strategy == "Dollar Cost Averaging":
                        amount = float(params.get("amount", 1000))
                        spy_fee_rate = slippage_bps / 10000.0
                        total_invested = 0
                        total_shares = 0
                        spy_equity_values = []

                        for i, (date, row) in enumerate(spy_sig_df.iterrows()):
                            if row["position_change"] > 0:  # Purchase day
                                effective_amount = amount * (1.0 - spy_fee_rate)
                                shares_bought = effective_amount / row["close"]
                                total_shares += shares_bought
                                total_invested += amount

                            # Current portfolio value
                            current_value = total_shares * row["close"]
                            spy_equity_values.append(current_value / total_invested if total_invested > 0 else 1.0)

                        spy_equity = pd.Series(spy_equity_values, index=spy_sig_df.index)

                        # Reindex to match original df index for comparison
                        spy_equity_aligned = spy_equity.reindex(df.index).ffill().bfill().fillna(1.0)
                        bh_equity = spy_equity_aligned

                    elif strategy == "Buy & Hold":
                        amount = float(params.get("amount", 10000))
                        spy_fee_rate = slippage_bps / 10000.0
                        initial_price = spy_prices.iloc[0]
                        spy_effective = amount * (1.0 - spy_fee_rate)
                        shares_bought = spy_effective / initial_price

                        # Calculate SPY Buy & Hold equity
                        spy_portfolio_values = shares_bought * spy_prices
                        spy_equity = spy_portfolio_values / amount  # Normalize vs nominal invested

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
        fee_rate = slippage_bps / 10000.0
        trades = []
        for date in df.index[df["position_change"] > 0.5]:
            px_in = prices.loc[date]
            effective_amount = amount * (1.0 - fee_rate)
            shares_bought = effective_amount / px_in
            # For DCA, we show each purchase as a trade from purchase to end
            px_out = prices.iloc[-1]  # Final price
            pnl = (px_out - px_in) * shares_bought
            ret_pct = (px_out / px_in - 1) * 100
            trades.append({
                "date_in": date,
                "date_out": df.index[-1],
                "cost_per_share": px_in,
                "total_cost": effective_amount,
                "pnl": pnl,
                "return_pct": ret_pct,
            })
        trades_df = pd.DataFrame(trades)
    elif strategy == "Buy & Hold":
        # For Buy & Hold, show one trade from start to end
        amount = float(params.get("amount", 10000))
        fee_rate = slippage_bps / 10000.0
        px_in = prices.iloc[0]
        px_out = prices.iloc[-1]
        effective_amount = amount * (1.0 - fee_rate)
        shares_bought = effective_amount / px_in
        pnl = (px_out - px_in) * shares_bought
        ret_pct = (px_out / px_in - 1) * 100
        trades = [{
            "date_in": df.index[0],
            "date_out": df.index[-1],
            "cost_per_share": px_in,
            "total_cost": effective_amount,
            "pnl": pnl,
            "return_pct": ret_pct,
        }]
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
            trades.append({
                "date_in": date,
                "date_out": df.index[-1],
                "cost_per_share": px_in,
                "total_cost": invest_amt,
                "pnl": pnl,
                "return_pct": ret_pct,
            })
        trades_df = pd.DataFrame(trades)
    else:
        # Original trade extraction logic for other strategies (RSI, Moving Average Crossover)
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
            ret_pct = (px_out / px_in - 1) * 100

            # Scale P/L to actual portfolio dollars when allocation is provided
            if allocation is not None and allocation > 0 and px_in > 0:
                shares_equiv = allocation / px_in
                dollar_pnl = shares_equiv * (px_out - px_in)
                total_cost_val = allocation
            else:
                dollar_pnl = px_out - px_in
                total_cost_val = px_in

            trades.append({
                "date_in": entry_date,
                "date_out": exit_date,
                "cost_per_share": px_in,
                "total_cost": total_cost_val,
                "pnl": dollar_pnl,
                "return_pct": ret_pct,
            })
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
