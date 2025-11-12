# Internationalization (i18n) Guide

This document explains how to add and maintain translation keys for the Trading Strategy Simulator.

## Overview

The app uses a lightweight JSON-based i18n system located in `i18n/`. Translations are stored as JSON files in `i18n/locales/` with support for English (`en`), Simplified Chinese (`zh-CN`), and Traditional Chinese (`zh-TW`).

## File Structure

```
i18n/
├── __init__.py
├── i18n.py          # Translation utility functions
└── locales/
    ├── en.json      # English translations
    ├── zh-CN.json   # Simplified Chinese translations
    └── zh-TW.json   # Traditional Chinese translations
```

## Translation Key Structure

Translation keys use dot notation to represent nested JSON structure. For example:

- `app.title` → `{"app": {"title": "..."}}`
- `sidebar.ticker_label` → `{"sidebar": {"ticker_label": "..."}}`
- `errors.fetch_error` → `{"errors": {"fetch_error": "..."}}`

## Using Translations in Code

### Basic Usage

```python
from i18n.i18n import t

# Simple translation
st.title(t("app.title"))

# Translation with parameters
st.error(t("errors.fetch_error", ticker="AAPL"))
```

### Language Management

```python
from i18n.i18n import t, set_language, get_lang

# Get current language
current_lang = get_lang()  # Returns "en", "zh-CN", or "zh-TW"

# Set language
set_language("zh-CN")
```

## Adding New Translation Keys

### Step 1: Add to English JSON

Edit `i18n/locales/en.json` and add your key following the existing structure:

```json
{
  "my_section": {
    "my_key": "My English Text"
  }
}
```

### Step 2: Add to Other Languages

Add the same key structure to:
- `i18n/locales/zh-CN.json` (Simplified Chinese)
- `i18n/locales/zh-TW.json` (Traditional Chinese)

### Step 3: Use in Code

```python
st.info(t("my_section.my_key"))
```

## Translation Key Naming Conventions

- Use lowercase with underscores: `my_section.my_key`
- Group related keys under common sections:
  - `app.*` - App-wide strings (title, disclaimer)
  - `sidebar.*` - Sidebar controls
  - `strategies.*` - Strategy names
  - `params.*` - Parameter labels
  - `errors.*` - Error messages
  - `charts.*` - Chart labels
  - `metrics.*` - Metric labels
  - `trades.*` - Trade table labels
  - `ui.*` - General UI messages

## String Formatting

For strings that need dynamic values, use Python's `.format()` syntax:

**In JSON:**
```json
{
  "errors": {
    "fetch_error": "Unable to fetch data for ticker '{ticker}'"
  }
}
```

**In Code:**
```python
t("errors.fetch_error", ticker="AAPL")
```

## Language Detection

The app automatically detects the browser language on first load:
- `zh-Hans*` or `zh-CN` → Simplified Chinese
- `zh-Hant*`, `zh-TW`, `zh-HK`, or `zh-MO` → Traditional Chinese
- Otherwise → English

The selected language persists in `st.session_state["lang"]` for the session.

## Best Practices

1. **Always add English first**: English is the fallback language
2. **Keep keys consistent**: Use the same key structure across all language files
3. **Use descriptive keys**: `errors.fetch_error` is better than `error1`
4. **Group related keys**: Keep related translations under the same section
5. **Test all languages**: Verify translations appear correctly in all supported languages
6. **Handle missing keys gracefully**: The system falls back to English if a key is missing

## Common Sections

### App Information
- `app.title` - Main app title
- `app.disclaimer` - Legal disclaimer

### Sidebar
- `sidebar.header` - Sidebar header
- `sidebar.ticker_label` - Stock ticker input label
- `sidebar.start_date` - Start date label
- `sidebar.end_date` - End date label
- `sidebar.strategy` - Strategy selector label
- `sidebar.run_button` - Run button text

### Strategies
- `strategies.dca` - Dollar Cost Averaging
- `strategies.buy_hold` - Buy & Hold
- `strategies.ma_crossover` - Moving Average Crossover
- `strategies.rsi` - RSI Strategy
- `strategies.new_car` - New Car strategy

### Charts
- `charts.price_title` - Price chart title (supports `{ticker}` parameter)
- `charts.equity_title` - Equity curve title
- `charts.price` - Price label
- `charts.buy` - Buy signal label
- `charts.sell` - Sell signal label
- `charts.strategy` - Strategy line label
- `charts.buy_hold` - Buy & Hold line label

### Metrics
- `metrics.total_return` - Total Return metric
- `metrics.sharpe_ratio` - Sharpe Ratio metric
- `metrics.max_drawdown` - Max Drawdown metric
- `metrics.buy_hold_return` - Buy & Hold Return metric

### Errors
- `errors.fetch_error` - Data fetch error (supports `{ticker}` parameter)
- `errors.verify_ticker` - Ticker verification message
- `errors.tip` - Helpful tip
- `errors.report_issue` - Issue reporting message
- `errors.no_data` - No data loaded (supports `{ticker}` parameter)
- `errors.backtest_error` - Backtest error (supports `{error}` parameter)
- `errors.try_again` - Retry message

## Troubleshooting

### Key Not Found

If a translation key is missing, the system will:
1. Try to find it in the current language's JSON
2. Fall back to English
3. If still not found, return the key itself

### Language Not Persisting

Ensure `set_language()` is called and `st.rerun()` is triggered after language change.

### Import Errors

Make sure the `i18n` package is in the Python path. The app uses:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i18n.i18n import t, set_language, get_lang
```

## Maintenance

When adding new features:
1. Identify all user-facing strings
2. Add translation keys to all three language files
3. Replace hardcoded strings with `t()` calls
4. Test in all three languages
5. Update this documentation if adding new sections

