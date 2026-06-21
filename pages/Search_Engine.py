"""Search Engine (TF-IDF + cosine similarity) — placeholder page."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("search_engine_title", icon="🔍")

st.title(f"🔍 {t('search_engine_title')}")
st.markdown(f"#### {t('search_engine_desc')}")
st.write("")

placeholder_banner()

# Search UI shown but not wired to any retrieval logic yet.
col1, col2 = st.columns([4, 1])
with col1:
    st.text_input(t("search_placeholder"), key="search_query", disabled=True)
with col2:
    st.button(t("search_button"), disabled=True, use_container_width=True)

render_footer()
