"""AI Student Assistant — placeholder page."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("ai_assistant_title", icon="🤖")

st.title(f"🤖 {t('ai_assistant_title')}")
st.markdown(f"#### {t('ai_assistant_desc')}")
st.write("")

placeholder_banner()

# Chat input shown but not wired to any model yet.
st.chat_input(t("ai_input_placeholder"), disabled=True)

render_footer()
