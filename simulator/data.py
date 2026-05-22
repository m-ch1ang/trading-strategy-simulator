import pandas as pd
import time
import json
import urllib.request
import urllib.parse

import streamlit as st

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None


def calculate_equal_weights(tickers: list, precision: int = 2) -> dict:
    """Return equal weights for tickers that sum to exactly 100.00.

    Uses integer-unit math to avoid floating-point drift.
    Remainder units are distributed to the last tickers so the first ticker
    is never systematically favored.
    """
    n = len(tickers)
    if n == 0:
        return {}
    total_units = 100 * (10 ** precision)  # 10000 for precision=2
    base_units = total_units // n
    remainder_units = total_units - base_units * n
    units = {tk: base_units for tk in tickers}
    # Distribute remainder to last tickers to avoid biasing the first
    for i in range(remainder_units):
        tk = tickers[n - 1 - (i % n)]
        units[tk] += 1
    factor = float(10 ** precision)
    return {tk: units[tk] / factor for tk in tickers}


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str, _version: str = "v5_yahoo_retry") -> tuple[pd.DataFrame, str]:
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

    def normalize_date_index(idx):
        """Return a timezone-naive DatetimeIndex for mixed provider outputs."""
        dt_idx = pd.to_datetime(idx)
        if getattr(dt_idx, "tz", None) is not None:
            dt_idx = dt_idx.tz_localize(None)
        return dt_idx

    # Primary: Yahoo Finance via yfinance (with retry to handle transient throttling/session issues)
    try:
        import yfinance as yf
        retries = 3
        retry_delay_seconds = 1.0
        yahoo_df = pd.DataFrame()

        for attempt in range(retries):
            try:
                yahoo_df = yf.download(
                    ticker.upper(),
                    start=start_dt,
                    end=end_dt,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

                # yfinance can return MultiIndex columns for some tickers/configs.
                if isinstance(yahoo_df.columns, pd.MultiIndex):
                    yahoo_df.columns = yahoo_df.columns.get_level_values(0)
                if not yahoo_df.empty and "Close" in yahoo_df.columns:
                    break
            except Exception:
                yahoo_df = pd.DataFrame()

            if attempt < retries - 1:
                time.sleep(retry_delay_seconds)

        if not yahoo_df.empty and "Close" in yahoo_df.columns:
            df = yahoo_df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            df.index = normalize_date_index(df.index)
            df.index.name = "date"
            return df, "yahoo"
    except Exception:
        pass

    # Secondary fallback: Yahoo Chart API (independent of yfinance/curl_cffi)
    try:
        period1 = int(pd.Timestamp(start_dt).timestamp())
        period2 = int(pd.Timestamp(end_dt).timestamp())
        chart_url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker.upper())}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history"
        )
        req = urllib.request.Request(chart_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            chart_raw = resp.read().decode("utf-8", errors="replace")
        chart_json = json.loads(chart_raw)
        chart_data = chart_json.get("chart", {})
        result = chart_data.get("result") or []
        if result:
            r0 = result[0]
            ts = r0.get("timestamp") or []
            quote = (((r0.get("indicators") or {}).get("quote") or [{}])[0]) or {}
            chart_df = pd.DataFrame({
                "date": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None),
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            })
            chart_df = chart_df.dropna(subset=["close"])
            if not chart_df.empty:
                chart_df = chart_df.set_index("date").sort_index()
                chart_df.index.name = "date"
                return chart_df, "yahoo"
    except Exception:
        pass

    # Fallback: Stooq via pandas-datareader
    try:
        if pdr is None:
            return pd.DataFrame(), "error"

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
                    df.index = normalize_date_index(df.index)
                    df.index.name = "date"
                    return df, "stooq"
            except Exception:
                continue
    except Exception:
        pass

    # If everything failed, return empty with error
    return pd.DataFrame(), "error"
