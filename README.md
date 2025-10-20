# Trading Strategy Simulator

Interactive Streamlit app to backtest simple trading strategies on stocks.

## Features
- Ticker input and date range selection
- Strategy picker: Moving Average Crossover, RSI Strategy, Dollar Cost Averaging, Buy & Hold, New Car
- Strategy-specific params
- Price chart with buy/sell markers
- Equity curve vs buy-and-hold
- Trades table
- Summary metrics: Total return, Sharpe, Max drawdown

### New Car Strategy
What if you had invested the money for a new car into a stock instead? This strategy simulates:
- Down payment invested immediately (either % of car price or an override amount)
- Recurring payments (weekly / biweekly / monthly) over a chosen term (12–60 months) invested into the stock instead of paying a loan
- Tracks total invested vs current value, gains, number of payments made, and completion status of the payment schedule
If the historical period ends before all scheduled payments, the strategy reports progress so far.

## Quick start

1. Create a virtual environment and install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app
```bash
streamlit run simulator/app.py
```

## Notes
- Primary Data source: Yahoo Finance via `yfinance`.
- Fallback Data source: Stooq via `pandas-datareader`.
- If ticker is not found, users are directed to verify ticker symbols at finance.yahoo.com
- This is a teaching/demo tool; not investment advice.
