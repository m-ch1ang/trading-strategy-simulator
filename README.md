# Trading Strategy Simulator

Interactive Streamlit app to backtest simple trading strategies on stocks.

## Features
- Ticker input and date range selection
- Strategy picker: Moving Average Crossover, RSI Strategy
- Strategy-specific params
- Price chart with buy/sell markers
- Equity curve vs buy-and-hold
- Trades table
- Summary metrics: Total return, Sharpe, Max drawdown

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
- Data source: Yahoo Finance via `yfinance`.
- This is a teaching/demo tool; not investment advice.
