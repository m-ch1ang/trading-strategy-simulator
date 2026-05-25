"""Shared footer and sidebar helpers for Streamlit pages."""

import streamlit as st

from i18n.i18n import t, set_language, get_lang


def render_language_selector() -> None:
    """Language selector for sidebar (used on Privacy page)."""
    label_to_code = {
        "English": "en",
        "繁體中文": "zh-TW",
        "简体中文": "zh-CN",
    }
    code_to_label = {v: k for k, v in label_to_code.items()}
    current = get_lang()
    sel = st.selectbox(
        t("language.label") + " / 語言 / 语言",
        options=list(label_to_code.keys()),
        index=list(label_to_code.keys()).index(code_to_label.get(current, "English")),
        key="footer_lang_select",
    )
    set_language(label_to_code[sel])


def render_footer(*, page: str = "home") -> None:
    """Bottom-of-page footer; navigates via st.switch_page + st.navigation pages."""
    st.divider()
    _, center, _ = st.columns([2, 1, 2])
    with center:
        if page == "privacy":
            label = t("footer.home")
            target = st.session_state.get("nav_page_home")
            button_key = "footer_nav_home"
        else:
            label = t("footer.privacy")
            target = st.session_state.get("nav_page_privacy")
            button_key = "footer_nav_privacy"

        if target is None:
            st.caption(label)
            return

        if st.button(label, key=button_key, type="tertiary"):
            st.switch_page(target)
