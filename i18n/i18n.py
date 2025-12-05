import json
import os
import streamlit as st

_SUPPORTED = ["en", "zh-CN", "zh-TW"]

# Always reload translations from disk so updates appear immediately
_CACHE = {}


def _load(lang):
    path = os.path.join(os.path.dirname(__file__), "locales", f"{lang}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    _CACHE[lang] = data
    return data


def get_lang():
    # priority: session -> detected -> en
    if "lang" in st.session_state and st.session_state["lang"] in _SUPPORTED:
        return st.session_state["lang"]
    # try browser locale once
    ctx = st.context if hasattr(st, "context") else None  # ignore if not available
    detected = None
    try:
        detected = st.runtime.scriptrun_ctx.get_client_language()  # if available
    except Exception:
        pass
    if detected:
        d = detected.lower()
        if "hans" in d or d.startswith("zh-cn"):
            st.session_state["lang"] = "zh-CN"
        elif "hant" in d or d.startswith("zh-tw") or d.startswith("zh-hk") or d.startswith("zh-mo"):
            st.session_state["lang"] = "zh-TW"
        else:
            st.session_state["lang"] = "en"
    else:
        st.session_state["lang"] = "en"
    return st.session_state["lang"]


def set_language(lang: str):
    if lang in _SUPPORTED:
        st.session_state["lang"] = lang


def _get_nested_value(data: dict, key_path: str, default: str = "") -> str:
    """Get nested value from dictionary using dot notation (e.g., 'app.title')."""
    keys = key_path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return str(value) if value is not None else default


def t(key: str, **kwargs):
    lang = get_lang()
    data = _load(lang)
    text = _get_nested_value(data, key, _get_nested_value(_load("en"), key, key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text

