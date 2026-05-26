import copy
import sys
import os
import pandas as pd
from functools import reduce

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i18n.i18n import t

from models import PortfolioResult
from data import load_data
from strategies import compute_signals
from backtest import backtest


def validate_portfolio_inputs(tickers: list, weights: dict) -> list:
    """Return list of user-facing error strings; empty list means valid."""
    errors = []
    if len(tickers) < 2:
        errors.append(t("portfolio.weight_error_min_tickers"))
        return errors
    seen = set()
    for ticker in tickers:
        if ticker in seen:
            errors.append(t("portfolio.weight_error_duplicate", ticker=ticker))
        seen.add(ticker)
    for ticker in tickers:
        if weights.get(ticker, 0) <= 0:
            errors.append(t("portfolio.weight_error_positive"))
            break
    weight_sum = sum(weights.get(tk, 0) for tk in tickers)
    if abs(weight_sum - 100.0) > 0.01:
        errors.append(t("portfolio.weight_error_sum", sum=f"{weight_sum:.1f}"))
    return errors


def compute_portfolio_params(base_params: dict, weight: float, total_capital: float, strategy: str) -> dict:
    """Return a deep copy of params with amount scaled by weight for dollar-based strategies.

    For Buy & Hold, total_capital is split by weight.
    For DCA, the user-entered amount per purchase is split by weight.
    """
    p = copy.deepcopy(base_params)
    if strategy == "Buy & Hold":
        p["amount"] = total_capital * weight
    elif strategy == "Dollar Cost Averaging":
        p["amount"] = p.get("amount", 0) * weight
    return p


def aggregate_portfolio_equity(
    ticker_results: dict,
    weights_normalized: dict,
    common_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute weighted-sum portfolio equity from per-ticker backtest results."""
    portfolio_equity = pd.Series(0.0, index=common_dates)
    for tk, result in ticker_results.items():
        bt = result["bt"]
        w = weights_normalized[tk]
        equity = bt["equity"].reindex(common_dates).ffill().bfill().fillna(1.0)
        portfolio_equity = portfolio_equity + w * equity
    portfolio_ret = portfolio_equity.pct_change().fillna(0.0)
    out = pd.DataFrame(index=common_dates)
    out["portfolio_equity"] = portfolio_equity
    out["portfolio_ret"] = portfolio_ret
    return out


def run_portfolio_backtest(
    tickers: list,
    weights_pct: dict,
    strategy: str,
    params: dict,
    slippage_bps: float,
    start: str,
    end: str,
    total_capital: float,
) -> PortfolioResult:
    """
    Orchestrate per-ticker load/signal/backtest, then aggregate into a portfolio.
    Raises ValueError with a user-facing message on fatal errors.
    """
    ticker_results = {}
    skipped_tickers = []

    for tk in tickers:
        try:
            df, source = load_data(tk, start, end)
            if df.empty:
                skipped_tickers.append(tk)
                continue
            ticker_params = compute_portfolio_params(
                params, weights_pct[tk] / 100.0, total_capital, strategy
            )
            sig_df = compute_signals(df, strategy, ticker_params)
            ticker_allocation = total_capital * weights_pct[tk] / 100.0
            bt, trades_df, dca_metrics, bh_metrics = backtest(
                sig_df, slippage_bps=slippage_bps, strategy=strategy, params=ticker_params,
                allocation=ticker_allocation,
            )
            if strategy == "Dollar Cost Averaging" and dca_metrics:
                dollar_allocation = dca_metrics.get("total_invested", 0.0)
            else:
                dollar_allocation = total_capital * weights_pct[tk] / 100.0
            ticker_results[tk] = {
                "bt": bt,
                "trades_df": trades_df,
                "dca_metrics": dca_metrics,
                "bh_metrics": bh_metrics,
                "source": source,
                "weight_pct": weights_pct[tk],
                "dollar_allocation": dollar_allocation,
                "sig_df": sig_df,
            }
        except Exception:
            skipped_tickers.append(tk)

    if not ticker_results:
        raise ValueError(t("errors.portfolio_all_failed"))

    successful = list(ticker_results.keys())
    total_w = sum(weights_pct[tk] for tk in successful)
    weights_normalized = {tk: weights_pct[tk] / total_w for tk in successful}

    all_indices = [ticker_results[tk]["bt"].index for tk in successful]
    common_dates = reduce(lambda a, b: a.intersection(b), all_indices)

    if len(common_dates) < 30:
        raise ValueError(t("portfolio.min_overlap_error", days=len(common_dates)))

    portfolio_df = aggregate_portfolio_equity(ticker_results, weights_normalized, common_dates)

    # SPY benchmark — DCA strategy uses SPY DCA equity; all others use price-return
    bh_equity = pd.Series(1.0, index=common_dates)
    try:
        spy_start = common_dates[0].strftime("%Y-%m-%d")
        spy_end = (common_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        spy_df, _ = load_data("SPY", spy_start, spy_end)
        if not spy_df.empty and "close" in spy_df.columns:
            if strategy == "Dollar Cost Averaging":
                # Compute SPY DCA equity with the same frequency and total periodic amount
                spy_aligned = spy_df[spy_df.index.isin(common_dates)].copy()
                if spy_aligned.empty:
                    spy_aligned = spy_df.reindex(common_dates).ffill().bfill()
                spy_sig_df = compute_signals(spy_aligned, strategy, params)
                spy_amount = float(params.get("amount", 1000))
                spy_total_invested = 0.0
                spy_total_shares = 0.0
                spy_equity_values = []
                for date in spy_sig_df.index:
                    row = spy_sig_df.loc[date]
                    if row["position_change"] > 0:
                        shares_bought = spy_amount / row["close"]
                        spy_total_shares += shares_bought
                        spy_total_invested += spy_amount
                    current_value = spy_total_shares * row["close"]
                    spy_equity_values.append(
                        current_value / spy_total_invested if spy_total_invested > 0 else 1.0
                    )
                spy_equity = pd.Series(spy_equity_values, index=spy_sig_df.index)
                bh_equity = spy_equity.reindex(common_dates).ffill().bfill().fillna(1.0)
            else:
                spy_prices = spy_df["close"].reindex(common_dates).ffill().bfill()
                spy_ret = spy_prices.pct_change().fillna(0.0)
                bh_equity = (1 + spy_ret).cumprod()
    except Exception:
        pass

    portfolio_df["bh_equity"] = bh_equity

    # Weighted buy & hold of portfolio tickers (each ticker's price-return equity, weighted)
    portfolio_bh_equity = pd.Series(0.0, index=common_dates)
    for tk, result in ticker_results.items():
        w = weights_normalized[tk]
        ticker_bh = result["bt"]["bh_equity"].reindex(common_dates).ffill().bfill().fillna(1.0)
        portfolio_bh_equity = portfolio_bh_equity + w * ticker_bh

    # For DCA, derive total_capital from the actual amounts invested across all tickers.
    if strategy == "Dollar Cost Averaging":
        total_capital = sum(
            r["dca_metrics"].get("total_invested", 0)
            for r in ticker_results.values()
            if r.get("dca_metrics")
        )

    return PortfolioResult(
        portfolio_equity=portfolio_df["portfolio_equity"],
        portfolio_ret=portfolio_df["portfolio_ret"],
        bh_equity=bh_equity,
        portfolio_bh_equity=portfolio_bh_equity,
        ticker_results=ticker_results,
        weights_normalized=weights_normalized,
        common_dates=common_dates,
        skipped_tickers=skipped_tickers,
        total_capital=total_capital,
    )
