"""BZU Student Intelligence Platform — Streamlit entry point.

Bilingual (English / Arabic) NLP application for ENCS5342.

This is the landing page of the multipage app. Navigate between features
using the sidebar; switch language with the sidebar language switcher.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.i18n.translations import (
    FEATURES,
    page_setup,
    render_footer,
    t,
)

lang = page_setup("landing_title", icon="🎓")

# --- Hero ------------------------------------------------------------------
st.title(f"🎓 {t('landing_title')}")
st.markdown(f"#### {t('landing_subtitle')}")
st.markdown(
    f'<span class="bzu-badge">{t("coming_soon")}</span>',
    unsafe_allow_html=True,
)
st.write("")
st.info(t("landing_get_started"))

st.write("")
st.subheader(t("features_heading"))

# --- Feature grid ----------------------------------------------------------
cols = st.columns(3)
for i, feature in enumerate(FEATURES):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="bzu-card">
                <div class="bzu-icon">{feature['icon']}</div>
                <h3>{t('nav_' + feature['key'])}</h3>
                <p style="color:#aab8d4;margin:0;">{t(feature['key'] + '_desc')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_footer()
