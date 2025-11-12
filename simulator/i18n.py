"""
Internationalization (i18n) helper module for Streamlit app.

Provides translation function t(key, lang) and Babel-based number/currency formatting.
"""
import yaml
from pathlib import Path
from typing import Any, Dict
from babel.numbers import format_currency, format_number, format_percent
from babel import Locale

# Locale mapping
LOCALE_MAP = {
    "en": "en_US",
    "zh-TW": "zh_TW",
    "zh-CN": "zh_CN"
}

# Currency mapping
CURRENCY_MAP = {
    "en": "USD",
    "zh-TW": "TWD",
    "zh-CN": "CNY"
}

# Load translations cache
_translations_cache: Dict[str, Dict[str, Any]] = {}


def _load_translations(lang: str) -> Dict[str, Any]:
    """Load translations from YAML file for the given language."""
    if lang in _translations_cache:
        return _translations_cache[lang]
    
    # Get the locales directory path
    locales_dir = Path(__file__).parent.parent / "locales"
    locale_file = locales_dir / f"{lang}.yml"
    
    if not locale_file.exists():
        # Fallback to English if locale file doesn't exist
        if lang != "en":
            return _load_translations("en")
        return {}
    
    try:
        with open(locale_file, "r", encoding="utf-8") as f:
            translations = yaml.safe_load(f)
            _translations_cache[lang] = translations
            return translations
    except Exception:
        # Fallback to English on error
        if lang != "en":
            return _load_translations("en")
        return {}


def _get_nested_value(data: Dict[str, Any], key_path: str, default: str = "") -> str:
    """Get nested value from dictionary using dot notation (e.g., 'app.title')."""
    keys = key_path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return str(value) if value is not None else default


def t(key: str, lang: str = "en", **kwargs) -> str:
    """
    Translate a key to the specified language.
    
    Args:
        key: Translation key in dot notation (e.g., 'app.title')
        lang: Language code ('en', 'zh-TW', 'zh-CN')
        **kwargs: Format arguments for string interpolation
        
    Returns:
        Translated string with format arguments applied
    """
    translations = _load_translations(lang)
    text = _get_nested_value(translations, key, key)
    
    # Apply format arguments if provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            # If formatting fails, return the text as-is
            pass
    
    return text


def format_currency_localized(amount: float, lang: str = "en") -> str:
    """
    Format currency amount using Babel based on language.
    
    Args:
        amount: Currency amount
        lang: Language code ('en', 'zh-TW', 'zh-CN')
        
    Returns:
        Formatted currency string
    """
    locale_str = LOCALE_MAP.get(lang, "en_US")
    currency = CURRENCY_MAP.get(lang, "USD")
    
    try:
        locale_obj = Locale.parse(locale_str.replace("_", "-"))
        return format_currency(amount, currency, locale=locale_obj)
    except Exception:
        # Fallback to simple formatting
        return f"${amount:,.2f}"


def format_number_localized(number: float, lang: str = "en", decimal_places: int = 2) -> str:
    """
    Format number using Babel based on language.
    
    Args:
        number: Number to format
        lang: Language code ('en', 'zh-TW', 'zh-CN')
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    locale_str = LOCALE_MAP.get(lang, "en_US")
    
    try:
        locale_obj = Locale.parse(locale_str.replace("_", "-"))
        return format_number(number, locale=locale_obj, decimal_quantization=False)
    except Exception:
        # Fallback to simple formatting
        return f"{number:,.{decimal_places}f}"


def format_percent_localized(value: float, lang: str = "en", decimal_places: int = 2) -> str:
    """
    Format percentage using Babel based on language.
    
    Args:
        value: Percentage value (e.g., 0.15 for 15%)
        lang: Language code ('en', 'zh-TW', 'zh-CN')
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    locale_str = LOCALE_MAP.get(lang, "en_US")
    
    try:
        locale_obj = Locale.parse(locale_str.replace("_", "-"))
        return format_percent(value, locale=locale_obj, decimal_quantization=False)
    except Exception:
        # Fallback to simple formatting
        return f"{value * 100:.{decimal_places}f}%"


def detect_browser_language() -> str:
    """
    Detect browser language from Streamlit query parameters or default to 'en'.
    
    Returns:
        Language code ('en', 'zh-TW', 'zh-CN')
    """
    try:
        import streamlit as st
        # Try to get language from query params
        query_params = st.query_params
        if "lang" in query_params:
            lang = query_params["lang"]
            if lang in ["en", "zh-TW", "zh-CN"]:
                return lang
        
        # Try to detect from browser Accept-Language header (if available)
        # Note: Streamlit doesn't expose this directly, so we default to 'en'
        # In a real deployment, you might get this from request headers
        return "en"
    except Exception:
        return "en"

