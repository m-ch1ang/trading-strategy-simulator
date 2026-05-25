"""Privacy Policy page content for st.navigation."""

import os
import sys
from pathlib import Path

import streamlit as st

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from i18n.i18n import t  # noqa: E402
from footer import render_footer, render_language_selector  # noqa: E402


def _load_privacy_markdown() -> str:
    path = Path(_repo_root) / "PRIVACY.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def render_privacy_page() -> None:
    with st.sidebar:
        render_language_selector()

    st.title(t("footer.privacy_page_title"))

    body = _load_privacy_markdown()
    if body:
        st.markdown(body)
    else:
        st.error(t("footer.privacy_missing"))

    render_footer(page="privacy")
