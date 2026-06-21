"""AI Student Assistant — placeholder page."""

import streamlit as st

from src.i18n.translations import page_header, placeholder_banner, render_footer, t

page_header("ai_assistant_title", "ai_assistant_desc", icon="🤖")
placeholder_banner()

# Chat input shown but not wired to any model yet.
st.chat_input(t("ai_input_placeholder"), disabled=True)

render_footer()
