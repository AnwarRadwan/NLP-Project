"""Home — dashboard overview."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("home_title", icon="🏠")

st.title(f"🏠 {t('home_title')}")
st.markdown(f"#### {t('home_welcome')}")
st.write(t("home_overview"))

st.write("")

# --- Top metrics (placeholder values) --------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(t("metric_documents"), "—")
c2.metric(t("metric_models"), "4")
c3.metric(t("metric_languages"), "2")
c4.metric(t("metric_features"), "9")

st.write("")
placeholder_banner()

render_footer()
