from dataclasses import dataclass


@dataclass
class PortfolioResult:
    portfolio_equity: object       # pd.Series
    portfolio_ret: object          # pd.Series
    bh_equity: object              # pd.Series — SPY benchmark (or SPY DCA for DCA strategy)
    portfolio_bh_equity: object    # pd.Series — weighted buy & hold of portfolio tickers
    ticker_results: dict
    weights_normalized: dict
    common_dates: object           # pd.DatetimeIndex
    skipped_tickers: list
    total_capital: float
