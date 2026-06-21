"""Search Engine (TF-IDF + cosine similarity) — placeholder page."""

import streamlit as st

from src.i18n.translations import page_header, placeholder_banner, render_footer, t

page_header("search_engine_title", "search_engine_desc", icon="🔍")
placeholder_banner()

# Search UI shown but not wired to any retrieval logic yet.
col1, col2 = st.columns([4, 1])
with col1:
    st.text_input(t("search_placeholder"), key="search_query", disabled=True,
                  label_visibility="collapsed")
with col2:
    st.button(t("search_button"), disabled=True, use_container_width=True)

render_footer()
