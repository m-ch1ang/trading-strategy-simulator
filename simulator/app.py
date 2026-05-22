import streamlit as st
import pandas as pd
import importlib
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i18n.i18n import t, set_language, get_lang

from data import load_data, calculate_equal_weights

_MARKDOWN_ESCAPE_CHARS = frozenset(r"\*_$[]()#+-{}.!`")


def escape_streamlit_markdown(text: str) -> str:
    """Escape characters Streamlit treats as Markdown/LaTeX (e.g. $ for math mode)."""
    return "".join(f"\\{c}" if c in _MARKDOWN_ESCAPE_CHARS else c for c in text)
from strategies import compute_signals
from backtest import backtest, sharpe_ratio, max_drawdown
from portfolio import (
    validate_portfolio_inputs,
    run_portfolio_backtest,
)


def resolve_plotly_module():
    """Resolve a Plotly module that definitely exposes Figure/Scatter."""
    # Preferred modern import path.
    try:
        go_mod = importlib.import_module("plotly.graph_objects")
        if hasattr(go_mod, "Figure") and hasattr(go_mod, "Scatter"):
            return go_mod
    except Exception:
        pass

    # Legacy alias import path.
    try:
        go_mod = importlib.import_module("plotly.graph_objs")
        if hasattr(go_mod, "Figure") and hasattr(go_mod, "Scatter"):
            return go_mod
    except Exception:
        pass

    # Self-heal for stale namespace packages in a long-running process.
    for mod_name in [m for m in list(sys.modules.keys()) if m == "plotly" or m.startswith("plotly.")]:
        sys.modules.pop(mod_name, None)
    importlib.invalidate_caches()
    go_mod = importlib.import_module("plotly.graph_objects")
    if hasattr(go_mod, "Figure") and hasattr(go_mod, "Scatter"):
        return go_mod
    raise ImportError("Plotly module loaded without Figure/Scatter support.")


go = resolve_plotly_module()


def main():
    st.set_page_config(page_title=t("app.title"), layout="wide")

    # Bug #5 (partial mitigation): Streamlit's base-web DatePicker opens its calendar
    # popup on any focus event, including Tab navigation. The `openOnFocus` prop is not
    # exposed by Streamlit's Python API, so a full fix requires a custom component or a
    # Streamlit framework update. The CSS below suppresses the auto-open visual artifact
    # for non-click focus events in supporting browsers by hiding the popover until the
    # user actively presses a key while the input is focused.
    st.markdown("""
<style>
/* Sidebar: 480px on wide screens; never wider than the viewport */
[data-testid="stSidebar"][aria-expanded="true"] {
    width: min(480px, 100vw) !important;
    min-width: min(480px, 100vw) !important;
    max-width: 100vw !important;
}
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: min(480px, 100vw) !important;
    max-width: 100vw !important;
}
/* Hide calendar popover that appears on Tab focus; keep visible on user interaction */
[data-baseweb="popover"][data-placement] {
    visibility: hidden;
    transition: visibility 0s 0.15s;
}
[data-baseweb="popover"][data-placement]:focus-within,
[data-baseweb="popover"][data-placement]:hover {
    visibility: visible;
    transition: none;
}
</style>
""", unsafe_allow_html=True)

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

        # --- MODE TOGGLE ---
        mode_single = t("portfolio.mode_single")
        mode_portfolio = t("portfolio.mode_portfolio")
        mode = st.radio(t("portfolio.mode_label"), [mode_single, mode_portfolio], horizontal=True)
        is_portfolio = (mode == mode_portfolio)

        # Bug #3: Reset strategy to a portfolio-compatible default when switching to portfolio
        # mode if "New Car" (single-ticker only) is currently selected.
        _strategy_key = "strategy_select"
        if is_portfolio and st.session_state.get(_strategy_key) == t("strategies.new_car"):
            st.session_state[_strategy_key] = t("strategies.dca")

        # --- TICKER / PORTFOLIO INPUTS ---
        if not is_portfolio:
            ticker = st.text_input(t("sidebar.ticker_label"), value="MSFT")
            parsed_tickers = [ticker.strip().upper()] if ticker.strip() else ["MSFT"]
            # Single-ticker mode includes all strategies
            strategy = st.selectbox(
                t("sidebar.strategy"),
                [t("strategies.dca"), t("strategies.buy_hold"), t("strategies.ma_crossover"), t("strategies.rsi"), t("strategies.new_car")],
                key=_strategy_key,
            )
            weights_pct = {parsed_tickers[0]: 100.0}
            total_capital = 10000.0
            portfolio_errors = []
        else:
            tickers_raw = st.text_area(
                t("portfolio.tickers_label"),
                value="AAPL, MSFT",
                height=80,
            )
            # Parse: split by comma or newline, strip, uppercase, deduplicate, keep order
            raw_parts = tickers_raw.replace(",", "\n").split("\n")
            parsed_tickers = list(dict.fromkeys(
                x.strip().upper() for x in raw_parts if x.strip()
            ))
            # Bug #3: "New Car" is a single-asset scenario — hide it entirely in portfolio mode
            strategy = st.selectbox(
                t("sidebar.strategy"),
                [t("strategies.dca"), t("strategies.buy_hold"), t("strategies.ma_crossover"), t("strategies.rsi")],
                key=_strategy_key,
            )

            if strategy != t("strategies.dca"):
                total_capital = st.number_input(
                    t("portfolio.total_capital_label"),
                    min_value=100.0,
                    value=10000.0,
                    step=1000.0,
                )
            else:
                total_capital = 0.0

            st.caption(t("portfolio.allocations_header"))

            # Bug #1 & #2: Auto-rebalance weights when ticker list changes.
            # Detect ticker change by comparing to previous render's ticker list.
            _prev_tickers = st.session_state.get("_portfolio_tickers_prev", None)
            _cur_tickers = tuple(parsed_tickers)
            if _cur_tickers != _prev_tickers and parsed_tickers:
                _new_weights = calculate_equal_weights(parsed_tickers)
                for _tk, _w in _new_weights.items():
                    st.session_state[f"weight_{_tk}"] = _w
                st.session_state["_portfolio_tickers_prev"] = _cur_tickers

            # Auto Equal Weights button — uses exact-total helper (Bug #2)
            if st.button(t("portfolio.auto_equal_weights")) and parsed_tickers:
                _new_weights = calculate_equal_weights(parsed_tickers)
                for _tk, _w in _new_weights.items():
                    st.session_state[f"weight_{_tk}"] = _w
                st.rerun()

            # Per-ticker weight inputs
            weights_pct = {}
            if parsed_tickers:
                for tk in parsed_tickers:
                    key = f"weight_{tk}"
                    default_val = float(st.session_state.get(key, 100.0 / len(parsed_tickers)))
                    w = st.number_input(
                        t("portfolio.weight_label", ticker=tk),
                        min_value=0.0,
                        max_value=100.0,
                        value=default_val,
                        step=1.0,
                        key=key,
                    )
                    weights_pct[tk] = w

            # Weight sum indicator
            weight_sum = sum(weights_pct.values())
            if abs(weight_sum - 100.0) <= 0.01:
                st.success(f"{t('portfolio.weight_sum_label')}: {weight_sum:.2f}% ✓")
            else:
                st.error(f"{t('portfolio.weight_sum_label')}: {weight_sum:.2f}% — {t('portfolio.weight_must_be_100')}")

            portfolio_errors = validate_portfolio_inputs(parsed_tickers, weights_pct)

        # --- DATES ---
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                t("sidebar.start_date"),
                value=datetime.today() - timedelta(days=365 * 3),
                min_value=datetime(1950, 1, 1),
            )
        with col2:
            end_date = st.date_input(
                t("sidebar.end_date"),
                value=datetime.today(),
                max_value=datetime.today(),
            )

        # Map translated strategy names back to internal names
        strategy_map = {
            t("strategies.dca"): "Dollar Cost Averaging",
            t("strategies.buy_hold"): "Buy & Hold",
            t("strategies.ma_crossover"): "Moving Average Crossover",
            t("strategies.rsi"): "RSI Strategy",
            t("strategies.new_car"): "New Car",
        }
        internal_strategy = strategy_map.get(strategy, strategy)

        # Safety guard: New Car is hidden in portfolio mode dropdown (Bug #3),
        # so this is False in normal flow but kept as a run-button backstop.
        new_car_in_portfolio = is_portfolio and internal_strategy == "New Car"

        # --- PARAMS ---
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
            if not is_portfolio:
                params["amount"] = st.number_input(t("params.initial_investment"), min_value=100, value=10000, step=100)
            else:
                params["amount"] = total_capital  # overridden per-ticker in run_portfolio_backtest
                st.caption(t("portfolio.amount_from_capital"))
            st.info(t("info.buy_hold_desc"))
        elif strategy == t("strategies.dca"):
            c1, c2 = st.columns(2)
            with c1:
                params["frequency"] = st.selectbox(t("params.buy_frequency"), ["Weekly", "Monthly", "Quarterly"], index=1)
            with c2:
                params["amount"] = st.number_input(t("params.dollar_amount"), min_value=100, value=1000, step=100)
        elif strategy == t("strategies.new_car"):
            # New Car is only reachable in single-ticker mode (hidden from portfolio dropdown)
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
            if freq == "Monthly":
                num_loan_payments = term_months
                rate_per_period = apr / 12
            elif freq == "Biweekly":
                num_loan_payments = int(round(term_months * 26 / 12))
                rate_per_period = apr / 26
            else:  # Weekly
                num_loan_payments = int(round(term_months * 52 / 12))
                rate_per_period = apr / 52
            num_loan_payments = max(num_loan_payments, 1)

            if rate_per_period > 0 and financed > 0:
                periodic_payment = financed * (rate_per_period * (1 + rate_per_period) ** num_loan_payments) / ((1 + rate_per_period) ** num_loan_payments - 1)
            else:
                periodic_payment = financed / num_loan_payments if num_loan_payments > 0 else 0

            params["_computed_periodic_payment"] = periodic_payment
            params["_computed_num_loan_payments"] = num_loan_payments
            params["_total_to_invest"] = dp + (periodic_payment * num_loan_payments)
            st.caption(escape_streamlit_markdown(t("info.car_payment_summary", dp=f"{dp:,.0f}", num_payments=num_loan_payments, payment=f"{periodic_payment:,.2f}", apr=f"{apr*100:.2f}", total=f"{dp + (periodic_payment * num_loan_payments):,.0f}")))

        slippage_bps = st.slider(t("sidebar.slippage_label"), min_value=0, max_value=50, value=0)

        run_disabled = is_portfolio and (bool(portfolio_errors) or new_car_in_portfolio)
        run = st.button(t("sidebar.run_button"), type="primary", disabled=run_disabled)

    # ------------------------------------------------------------------ #
    # RUN BLOCK                                                            #
    # ------------------------------------------------------------------ #
    if run:
        start_str = start_date.isoformat()
        end_str = (end_date + timedelta(days=1)).isoformat()

        # ---- SINGLE TICKER MODE ----
        if not is_portfolio:
            with st.spinner(t("ui.loading")):
                try:
                    df, data_source = load_data(ticker.strip().upper(), start_str, end_str)

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

            # Metrics
            total_return = float(bt["equity"].iloc[-1]) - 1
            sr = sharpe_ratio(bt["strategy_ret"])
            mdd = max_drawdown(bt["equity"])

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

                loan_pmts = dca_metrics.get('loan_payments_made', 0)
                total_loan = dca_metrics.get('num_loan_payments', 0)
                payments_completed = dca_metrics.get('payments_completed', 0)
                total_expected = dca_metrics.get('total_expected', 0)
                status_text = t("new_car_metrics.completed") if dca_metrics.get('completed') else t("new_car_metrics.in_progress")

                apr_display = params.get('apr', None)
                apr_text = f" | APR: {apr_display:.2f}%" if apr_display is not None else ""

                st.info(t("new_car_metrics.payment_progress", completed=payments_completed, total=total_expected, status=status_text))
                st.caption(escape_streamlit_markdown(t("new_car_metrics.payment_details", dp=f"{dca_metrics['down_payment']:,.2f}", loan_pmts=loan_pmts, total_loan=total_loan, payment=f"{dca_metrics['periodic_payment']:,.2f}", apr=apr_text, shares=f"{dca_metrics['total_shares']:.4f}")))

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
                if "cost_per_share" in tshow.columns:
                    tshow["cost_per_share"] = tshow["cost_per_share"].map(lambda x: f"${float(x):,.2f}")
                if "total_cost" in tshow.columns:
                    tshow["total_cost"] = tshow["total_cost"].map(lambda x: f"${float(x):,.2f}")
                tshow["pnl"] = tshow["pnl"].map(lambda x: f"${float(x):,.2f}")
                tshow["return_pct"] = tshow["return_pct"].map(lambda x: f"{float(x):.2f}%")
                tshow = tshow.rename(columns={
                    "date_in": t("trades.entry_date"),
                    "date_out": t("trades.exit_date"),
                    "cost_per_share": t("trades.cost_per_share"),
                    "total_cost": t("trades.total_cost"),
                    "pnl": t("trades.pnl"),
                    "return_pct": t("trades.return_pct")
                })
                ordered_cols = [
                    t("trades.entry_date"),
                    t("trades.exit_date"),
                    t("trades.cost_per_share"),
                    t("trades.total_cost"),
                    t("trades.pnl"),
                    t("trades.return_pct"),
                ]
                tshow = tshow[[col for col in ordered_cols if col in tshow.columns]]
                st.dataframe(tshow, use_container_width=True, hide_index=True)
            else:
                st.info(t("trades.no_trades"))

        # ---- PORTFOLIO MODE ----
        else:
            with st.spinner(t("ui.loading")):
                try:
                    portfolio_result = run_portfolio_backtest(
                        tickers=parsed_tickers,
                        weights_pct=weights_pct,
                        strategy=internal_strategy,
                        params=params,
                        slippage_bps=float(slippage_bps),
                        start=start_str,
                        end=end_str,
                        total_capital=float(total_capital),
                    )
                except ValueError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.error(t("errors.backtest_error", error=str(e)))
                    st.error(t("errors.try_again"))
                    return

            # Warnings for skipped tickers
            if portfolio_result.skipped_tickers:
                skipped_str = ", ".join(portfolio_result.skipped_tickers)
                eff_weights = ", ".join(
                    f"{tk}: {portfolio_result.weights_normalized[tk]*100:.1f}%"
                    for tk in portfolio_result.ticker_results
                )
                st.warning(t("portfolio.skipped_tickers_warning", tickers=skipped_str))
                st.caption(t("portfolio.effective_weights_caption", weights=eff_weights))

            # Data source info
            sources = set(v["source"] for v in portfolio_result.ticker_results.values())
            src_str = ", ".join(sorted(sources))
            st.success(f"Portfolio loaded — sources: {src_str} | {len(portfolio_result.common_dates)} common trading days")

            # ---- PORTFOLIO METRICS (4-column) ----
            port_total_return = float(portfolio_result.portfolio_equity.iloc[-1]) - 1
            port_sr = sharpe_ratio(portfolio_result.portfolio_ret)
            port_mdd = max_drawdown(portfolio_result.portfolio_equity)
            spy_return = float(portfolio_result.bh_equity.iloc[-1]) - 1

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t("metrics.portfolio_total_return"), f"{port_total_return*100:.2f}%")
            c2.metric(t("metrics.sharpe_ratio"), f"{port_sr:.2f}")
            c3.metric(t("metrics.max_drawdown"), f"{port_mdd*100:.2f}%")
            c4.metric("SPY " + t("metrics.buy_hold_return"), f"{spy_return*100:.2f}%")

            # ---- PORTFOLIO DOLLAR METRICS (3-column) ----
            if internal_strategy == "Dollar Cost Averaging":
                # Sum per-ticker actuals from dca_metrics
                port_total_invested = sum(
                    r["dca_metrics"].get("total_invested", 0)
                    for r in portfolio_result.ticker_results.values()
                    if r.get("dca_metrics")
                )
                port_total_value = sum(
                    r["dca_metrics"].get("total_value", 0)
                    for r in portfolio_result.ticker_results.values()
                    if r.get("dca_metrics")
                )
                port_total_gain = port_total_value - port_total_invested
                gain_pct = (port_total_gain / port_total_invested * 100) if port_total_invested else 0.0
            else:
                port_total_invested = portfolio_result.total_capital
                port_total_value = portfolio_result.total_capital * float(portfolio_result.portfolio_equity.iloc[-1])
                port_total_gain = port_total_value - port_total_invested
                gain_pct = (port_total_gain / port_total_invested * 100) if port_total_invested else 0.0

            dm1, dm2, dm3 = st.columns(3)
            dm1.metric(t("dca_metrics.total_invested"), f"${port_total_invested:,.2f}")
            dm2.metric(t("dca_metrics.total_value"), f"${port_total_value:,.2f}")
            dm3.metric(t("dca_metrics.total_gain"), f"${port_total_gain:,.2f}", delta=f"{gain_pct:+.2f}%")

            # ---- PORTFOLIO EQUITY CURVE ----
            st.subheader(t("charts.portfolio_equity_title"))
            port_eq_fig = go.Figure()
            port_eq_fig.add_trace(go.Scatter(
                x=portfolio_result.common_dates,
                y=portfolio_result.portfolio_equity,
                mode="lines",
                name=t("charts.portfolio_strategy"),
            ))
            port_eq_fig.add_trace(go.Scatter(
                x=portfolio_result.common_dates,
                y=portfolio_result.bh_equity,
                mode="lines",
                name=t("charts.spy_buy_hold"),
            ))
            port_eq_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(port_eq_fig, use_container_width=True)

            # ---- PER-TICKER EQUITY CURVES ----
            st.subheader(t("charts.per_ticker_equity_title"))
            ticker_eq_fig = go.Figure()
            for tk, result in portfolio_result.ticker_results.items():
                bt = result["bt"]
                w = portfolio_result.weights_normalized[tk]
                equity = bt["equity"].reindex(portfolio_result.common_dates).ffill().bfill().fillna(1.0)
                ticker_eq_fig.add_trace(go.Scatter(
                    x=portfolio_result.common_dates,
                    y=equity,
                    mode="lines",
                    name=f"{tk} ({w*100:.1f}%)",
                ))
            ticker_eq_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(ticker_eq_fig, use_container_width=True)

            # ---- TRADES TABLE ----
            st.subheader(t("trades.title"))
            all_trades = []
            for tk, result in portfolio_result.ticker_results.items():
                tdf = result["trades_df"].copy()
                if not tdf.empty:
                    tdf.insert(0, t("trades.ticker_col"), tk)
                    all_trades.append(tdf)

            if all_trades:
                combined_trades = pd.concat(all_trades, ignore_index=True)
                # Ticker filter
                all_tickers_list = list(portfolio_result.ticker_results.keys())
                selected_tickers = st.multiselect(
                    t("trades.filter_label"),
                    all_tickers_list,
                    default=all_tickers_list,
                )
                ticker_col_name = t("trades.ticker_col")
                if selected_tickers and ticker_col_name in combined_trades.columns:
                    combined_trades = combined_trades[combined_trades[ticker_col_name].isin(selected_tickers)]

                tshow = combined_trades.copy()
                tshow["date_in"] = pd.to_datetime(tshow["date_in"]).dt.strftime("%Y-%m-%d")
                tshow["date_out"] = pd.to_datetime(tshow["date_out"]).dt.strftime("%Y-%m-%d")
                if "cost_per_share" in tshow.columns:
                    tshow["cost_per_share"] = tshow["cost_per_share"].map(lambda x: f"${float(x):,.2f}")
                if "total_cost" in tshow.columns:
                    tshow["total_cost"] = tshow["total_cost"].map(lambda x: f"${float(x):,.2f}")
                tshow["pnl"] = tshow["pnl"].map(lambda x: f"${float(x):,.2f}")
                tshow["return_pct"] = tshow["return_pct"].map(lambda x: f"{float(x):.2f}%")
                tshow = tshow.rename(columns={
                    "date_in": t("trades.entry_date"),
                    "date_out": t("trades.exit_date"),
                    "cost_per_share": t("trades.cost_per_share"),
                    "total_cost": t("trades.total_cost"),
                    "pnl": t("trades.pnl"),
                    "return_pct": t("trades.return_pct"),
                })
                ordered_cols = [
                    t("trades.ticker_col"),
                    t("trades.entry_date"),
                    t("trades.exit_date"),
                    t("trades.cost_per_share"),
                    t("trades.total_cost"),
                    t("trades.pnl"),
                    t("trades.return_pct"),
                ]
                tshow = tshow[[col for col in ordered_cols if col in tshow.columns]]
                st.dataframe(tshow, use_container_width=True, hide_index=True)
            else:
                st.info(t("trades.no_trades"))

            # ---- PER-TICKER DETAIL EXPANDER ----
            with st.expander(t("portfolio_detail.title"), expanded=False):
                detail_rows = []
                for tk, result in portfolio_result.ticker_results.items():
                    bt = result["bt"]
                    w = portfolio_result.weights_normalized[tk]
                    ticker_return = float(bt["equity"].iloc[-1]) - 1
                    ticker_sr = sharpe_ratio(bt["strategy_ret"])
                    ticker_mdd = max_drawdown(bt["equity"])
                    detail_rows.append({
                        t("portfolio_detail.ticker_col"): tk,
                        t("portfolio_detail.weight_col"): f"{w*100:.1f}%",
                        t("portfolio_detail.allocation_col"): f"${result['dollar_allocation']:,.2f}",
                        t("portfolio_detail.return_col"): f"{ticker_return*100:.2f}%",
                        t("portfolio_detail.sharpe_col"): f"{ticker_sr:.2f}",
                        t("portfolio_detail.mdd_col"): f"{ticker_mdd*100:.2f}%",
                        t("portfolio_detail.source_col"): result["source"],
                    })
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    else:
        st.info(t("ui.waiting"))


if __name__ == "__main__":
    main()
