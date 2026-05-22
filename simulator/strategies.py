import pandas as pd
import numpy as np


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
        df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

        # Calculate position changes
        df["position_change"] = df["signal"].diff().fillna(0)
    elif strategy == "Buy & Hold":
        amount = float(params.get("amount", 10000))
        # Calculate shares bought with initial investment
        initial_price = df["close"].iloc[0]
        shares_bought = amount / initial_price

        # Always hold the same number of shares
        df["signal"] = shares_bought  # Number of shares held
        df["position_change"] = 0.0  # No position changes after initial buy (float for fractional shares)
        # Set initial buy signal
        df.iloc[0, df.columns.get_loc("position_change")] = shares_bought  # Initial buy
        df["bh_amount"] = amount  # Track initial investment amount
    elif strategy == "Dollar Cost Averaging":
        frequency = params.get("frequency", "Monthly")
        amount = float(params.get("amount", 1000))

        # Initialize signals
        df["signal"] = 0.0
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
