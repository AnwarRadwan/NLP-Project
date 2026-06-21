"""Analytics Dashboard — placeholder page."""

import streamlit as st

from src.i18n.translations import page_header, placeholder_banner, render_footer, t

page_header("analytics_title", "analytics_desc", icon="📊")
placeholder_banner()

# Chart placeholders (empty cards, no real data yet).
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f'<div class="feature-card"><div class="feature-icon">📈</div>'
        f'<p>{t("coming_soon")}</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="feature-card"><div class="feature-icon">🥧</div>'
        f'<p>{t("coming_soon")}</p></div>',
        unsafe_allow_html=True,
    )

render_footer()
