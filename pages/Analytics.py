"""Analytics Dashboard — placeholder page."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("analytics_title", icon="📊")

st.title(f"📊 {t('analytics_title')}")
st.markdown(f"#### {t('analytics_desc')}")
st.write("")

placeholder_banner()

# Chart placeholders (empty containers, no real data yet).
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f'<div class="bzu-card"><h3>📈</h3><p style="color:#aab8d4;">{t("coming_soon")}</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="bzu-card"><h3>🥧</h3><p style="color:#aab8d4;">{t("coming_soon")}</p></div>',
        unsafe_allow_html=True,
    )

render_footer()
