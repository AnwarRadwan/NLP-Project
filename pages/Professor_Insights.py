"""Professor & Course Insights — placeholder page."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("professor_insights_title", icon="👨‍🏫")

st.title(f"👨‍🏫 {t('professor_insights_title')}")
st.markdown(f"#### {t('professor_insights_desc')}")
st.write("")

placeholder_banner()

left, right = st.columns([2, 1])
with left:
    st.markdown(
        f'<div class="bzu-card"><h3>{t("professor_insights_title")}</h3>'
        f'<p style="color:#aab8d4;">{t("placeholder_notice")}</p></div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f'<div class="bzu-card"><span class="bzu-badge">{t("coming_soon")}</span></div>',
        unsafe_allow_html=True,
    )

render_footer()
